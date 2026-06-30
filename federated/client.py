from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import WindowedData
from evaluation.metrics import all_metrics


class RULFaultDataset(Dataset):
    def __init__(self, data: WindowedData) -> None:
        self.x = torch.from_numpy(data.x)
        self.rul = torch.from_numpy(data.rul)
        self.fault = torch.from_numpy(data.fault)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.rul[index], self.fault[index]


def multitask_loss(
    rul_pred: torch.Tensor,
    fault_logit: torch.Tensor,
    rul_true: torch.Tensor,
    fault_true: torch.Tensor,
    fault_loss_weight: float,
    fault_pos_weight: float | None = None,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(rul_pred, rul_true)
    pos_weight = None
    if fault_pos_weight is not None:
        pos_weight = torch.tensor(fault_pos_weight, dtype=fault_logit.dtype, device=fault_logit.device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        fault_logit,
        fault_true,
        pos_weight=pos_weight,
    )
    return mse + fault_loss_weight * bce


def train_model(
    model: torch.nn.Module,
    data: WindowedData,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    fault_loss_weight: float,
    device: torch.device,
    fault_pos_weight: float | None = None,
) -> float:
    if len(data) == 0:
        return float("nan")

    model.to(device)
    model.train()
    loader = DataLoader(RULFaultDataset(data), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    total_count = 0

    for _ in range(epochs):
        for x, rul, fault in loader:
            x = x.to(device)
            rul = rul.to(device)
            fault = fault.to(device)
            optimizer.zero_grad()
            rul_pred, fault_logit = model(x)
            loss = multitask_loss(
                rul_pred,
                fault_logit,
                rul,
                fault,
                fault_loss_weight,
                fault_pos_weight=fault_pos_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_count += int(x.shape[0])

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data: WindowedData,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    loader = DataLoader(RULFaultDataset(data), batch_size=batch_size, shuffle=False)
    rul_preds = []
    fault_scores = []
    rul_true = []
    fault_true = []

    for x, rul, fault in loader:
        x = x.to(device)
        rul_pred, fault_logit = model(x)
        rul_preds.append(rul_pred.cpu().numpy())
        fault_scores.append(torch.sigmoid(fault_logit).cpu().numpy())
        rul_true.append(rul.numpy())
        fault_true.append(fault.numpy())

    if not rul_preds:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty

    return (
        np.concatenate(rul_true),
        np.concatenate(rul_preds),
        np.concatenate(fault_true),
        np.concatenate(fault_scores),
    )


def evaluate_model(
    model: torch.nn.Module,
    data: WindowedData,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    rul_true, rul_pred, fault_true, fault_score = predict(model, data, batch_size, device)
    if len(rul_true) == 0:
        return {}
    return all_metrics(rul_true, rul_pred, fault_true, fault_score)
