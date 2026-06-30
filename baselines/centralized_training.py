from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import WindowedData
from federated.client import evaluate_model, train_model
from federated.server import clone_model


def concatenate_client_data(clients: dict[int, dict]) -> WindowedData:
    return WindowedData(
        x=np.concatenate([client["train"].x for client in clients.values()], axis=0),
        rul=np.concatenate([client["train"].rul for client in clients.values()], axis=0),
        fault=np.concatenate([client["train"].fault for client in clients.values()], axis=0),
        unit=np.concatenate([client["train"].unit for client in clients.values()], axis=0),
    )


def run_centralized(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    test_data,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    cfg = config["centralized"]
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = clone_model(base_model)
    train_data = concatenate_client_data(clients)
    train_loss = train_model(
        model,
        train_data,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        fault_loss_weight=cfg["fault_loss_weight"],
        device=device,
    )
    metrics = evaluate_model(model, test_data, cfg["batch_size"], device)
    row = {
        "experiment": "centralized",
        "client_id": "global",
        "num_train_windows": len(train_data),
        "train_loss": train_loss,
    }
    row.update(metrics)
    results = pd.DataFrame([row])
    results.to_csv(out_dir / "centralized_results.csv", index=False)
    return results
