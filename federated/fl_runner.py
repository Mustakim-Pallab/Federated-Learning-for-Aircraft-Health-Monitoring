from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from federated.client import evaluate_model, train_model
from federated.server import clone_model, fedavg_state_dicts


def run_federated(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    validation_data,
    test_data,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    fl_cfg = config["federated"]
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    global_model = clone_model(base_model)
    records = []
    best_state = None
    best_rmse = float("inf")

    for round_id in range(1, fl_cfg["rounds"] + 1):
        client_states = []
        sample_counts = []
        train_losses = []

        for client_id, client in clients.items():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())
            loss = train_model(
                local_model,
                client["train"],
                epochs=fl_cfg["local_epochs"],
                batch_size=fl_cfg["batch_size"],
                learning_rate=fl_cfg["learning_rate"],
                fault_loss_weight=fl_cfg["fault_loss_weight"],
                device=device,
            )
            client_states.append(local_model.state_dict())
            sample_counts.append(len(client["train"]))
            train_losses.append(loss)

        global_model.load_state_dict(fedavg_state_dicts(client_states, sample_counts))

        val_metrics = evaluate_model(global_model, validation_data, fl_cfg["batch_size"], device)
        test_metrics = evaluate_model(global_model, test_data, fl_cfg["batch_size"], device)
        if val_metrics and val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        row = {
            "experiment": "federated",
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        records.append(row)

    if best_state is not None:
        global_model.load_state_dict(best_state)

    results = pd.DataFrame(records)
    results.to_csv(out_dir / "federated_rounds.csv", index=False)
    return global_model, results
