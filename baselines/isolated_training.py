from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from federated.client import evaluate_model, train_model
from federated.server import clone_model


def run_isolated(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    test_data,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    cfg = config["isolated"]
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for client_id, client in clients.items():
        model = clone_model(base_model)
        train_loss = train_model(
            model,
            client["train"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            fault_loss_weight=cfg["fault_loss_weight"],
            device=device,
        )
        metrics = evaluate_model(model, test_data, cfg["batch_size"], device)
        row = {
            "experiment": "isolated",
            "client_id": client_id,
            "num_train_windows": len(client["train"]),
            "train_loss": train_loss,
        }
        row.update(metrics)
        records.append(row)

    results = pd.DataFrame(records)
    results.to_csv(out_dir / "isolated_results.csv", index=False)
    return results
