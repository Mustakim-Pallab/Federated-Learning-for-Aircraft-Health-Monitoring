from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.preprocessing import (
    DEFAULT_SENSORS,
    WindowedData,
    apply_scaler,
    fit_scaler,
    load_cmapss,
    make_windows,
    maybe_download_cmapss,
    partition_units,
    split_units,
    subset_units,
)
from evaluation.metrics import all_metrics
from federated.client import RULFaultDataset, evaluate_model
from federated.fl_runner import run_federated
from models.dual_head_model import build_model
from models.heterogeneous_sensor_model import build_heterogeneous_sensor_model


def _client_sensor_sets(config: dict) -> dict[int, list[str]]:
    sensor_sets = config["rq1"]["client_sensor_sets"]
    return {int(client_id): list(sensors) for client_id, sensors in sensor_sets.items()}


def prepare_variable_sensor_clients(config: dict) -> dict:
    ds_cfg = config["dataset"]
    fl_cfg = config["federated"]

    if ds_cfg.get("download", False):
        maybe_download_cmapss(ds_cfg["url"], ds_cfg["data_dir"])

    all_sensors = ds_cfg.get("sensors") or DEFAULT_SENSORS
    client_sensor_sets = _client_sensor_sets(config)
    train_df, test_df = load_cmapss(
        data_dir=ds_cfg["data_dir"],
        subset=ds_cfg["name"],
        rul_cap=ds_cfg["rul_cap"],
        fault_threshold=ds_cfg["fault_threshold"],
    )

    train_units, val_units = split_units(
        sorted(train_df["unit"].unique().tolist()),
        validation_fraction=ds_cfg["validation_engine_fraction"],
        seed=config["seed"],
    )
    train_only = subset_units(train_df, train_units)
    val_only = subset_units(train_df, val_units)

    scaler = fit_scaler(train_only, all_sensors)
    train_only = apply_scaler(train_only, all_sensors, scaler)
    val_only = apply_scaler(val_only, all_sensors, scaler)
    test_df = apply_scaler(test_df, all_sensors, scaler)

    client_units = partition_units(
        train_units,
        num_clients=fl_cfg["num_clients"],
        seed=config["seed"],
        strategy=fl_cfg.get("partition_strategy", "iid"),
    )

    clients = {}
    validation_by_client = {}
    test_by_client = {}
    for client_id, units in client_units.items():
        sensors = client_sensor_sets[client_id]
        clients[client_id] = {
            "train": make_windows(
                subset_units(train_only, units),
                sensors,
                ds_cfg["window_size"],
                ds_cfg["stride"],
            ),
            "units": units,
            "sensors": sensors,
        }
        validation_by_client[client_id] = make_windows(
            val_only,
            sensors,
            ds_cfg["window_size"],
            ds_cfg["stride"],
        )
        test_by_client[client_id] = make_windows(
            test_df,
            sensors,
            ds_cfg["window_size"],
            ds_cfg["stride"],
        )

    return {
        "clients": clients,
        "validation_by_client": validation_by_client,
        "test_by_client": test_by_client,
        "client_sensor_sets": client_sensor_sets,
        "client_input_sizes": {
            client_id: len(sensors) for client_id, sensors in client_sensor_sets.items()
        },
        "all_sensors": all_sensors,
        "scaler": scaler,
    }


def train_adapter_model(
    model: torch.nn.Module,
    client_id: int,
    data: WindowedData,
    config: dict,
    device: torch.device,
) -> float:
    if len(data) == 0:
        return float("nan")

    fl_cfg = config["federated"]
    model.to(device)
    model.train()
    loader = DataLoader(RULFaultDataset(data), batch_size=fl_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=fl_cfg["learning_rate"])
    total_loss = 0.0
    total_count = 0

    for _ in range(fl_cfg["local_epochs"]):
        for x, rul, fault in loader:
            x = x.to(device)
            rul = rul.to(device)
            fault = fault.to(device)
            optimizer.zero_grad()
            rul_pred, fault_logit = model(x, client_id)
            mse = nn.functional.mse_loss(rul_pred, rul)
            bce = nn.functional.binary_cross_entropy_with_logits(fault_logit, fault)
            loss = mse + fl_cfg["fault_loss_weight"] * bce
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_count += int(x.shape[0])

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_adapter_model(
    model: torch.nn.Module,
    client_id: int,
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
        rul_pred, fault_logit = model(x, client_id)
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


def evaluate_adapter_by_client(
    model: torch.nn.Module,
    data_by_client: dict[int, WindowedData],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    records = []
    all_rul_true = []
    all_rul_pred = []
    all_fault_true = []
    all_fault_score = []

    for client_id, data in data_by_client.items():
        rul_true, rul_pred, fault_true, fault_score = predict_adapter_model(
            model,
            client_id,
            data,
            batch_size,
            device,
        )
        metrics = all_metrics(rul_true, rul_pred, fault_true, fault_score)
        records.append({"client_id": client_id, **metrics})
        all_rul_true.append(rul_true)
        all_rul_pred.append(rul_pred)
        all_fault_true.append(fault_true)
        all_fault_score.append(fault_score)

    overall_metrics = all_metrics(
        np.concatenate(all_rul_true),
        np.concatenate(all_rul_pred),
        np.concatenate(all_fault_true),
        np.concatenate(all_fault_score),
    )
    per_client = pd.DataFrame(records)
    metric_cols = [
        "rmse",
        "mae",
        "nasa_score",
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    macro_metrics = {
        column: float(per_client[column].mean())
        for column in metric_cols
        if column in per_client
    }
    return per_client, overall_metrics, macro_metrics


def shared_state_keys(model: torch.nn.Module) -> list[str]:
    return [key for key in model.state_dict().keys() if not key.startswith("adapters.")]


def aggregate_shared_state(
    client_states: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    keys: list[str],
) -> dict[str, torch.Tensor]:
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Total sample count must be positive")

    averaged = {}
    for key in keys:
        first_value = client_states[0][key]
        if not torch.is_floating_point(first_value):
            averaged[key] = first_value.detach().cpu().clone()
            continue

        value = None
        for state, count in zip(client_states, sample_counts):
            weighted = state[key].detach().cpu() * (count / total)
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged


def run_variable_sensor_adapter_fl(
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, pd.DataFrame]:
    prepared = prepare_variable_sensor_clients(config)
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_heterogeneous_sensor_model(prepared["client_input_sizes"], config)
    shared_keys = shared_state_keys(model)
    records = []
    best_state = None
    best_val_rmse = float("inf")

    client_metadata = pd.DataFrame(
        [
            {
                "client_id": client_id,
                "num_sensors": len(client["sensors"]),
                "sensors": " ".join(client["sensors"]),
                "num_train_windows": len(client["train"]),
            }
            for client_id, client in prepared["clients"].items()
        ]
    )
    client_metadata.to_csv(out_dir / "rq1_client_sensor_sets.csv", index=False)

    for round_id in range(1, config["federated"]["rounds"] + 1):
        local_states = []
        sample_counts = []
        train_losses = []
        adapter_states = {}

        for client_id, client in prepared["clients"].items():
            local_model = copy.deepcopy(model)
            loss = train_adapter_model(local_model, client_id, client["train"], config, device)
            local_state = local_model.state_dict()
            local_states.append(local_state)
            sample_counts.append(len(client["train"]))
            train_losses.append(loss)
            prefix = f"adapters.{client_id}."
            adapter_states[client_id] = {
                key: value.detach().cpu().clone()
                for key, value in local_state.items()
                if key.startswith(prefix)
            }

        next_state = model.state_dict()
        next_state.update(aggregate_shared_state(local_states, sample_counts, shared_keys))
        for state in adapter_states.values():
            next_state.update(state)
        model.load_state_dict(next_state)

        _, val_metrics, _ = evaluate_adapter_by_client(
            model,
            prepared["validation_by_client"],
            config["federated"]["batch_size"],
            device,
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        row = {
            "experiment": "rq1_variable_sensor_adapter",
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        records.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)

    rounds = pd.DataFrame(records)
    rounds.to_csv(out_dir / "rq1_variable_sensor_adapter_rounds.csv", index=False)
    per_client, overall, macro_metrics = evaluate_adapter_by_client(
        model,
        prepared["test_by_client"],
        config["federated"]["batch_size"],
        device,
    )
    per_client.to_csv(out_dir / "rq1_variable_sensor_adapter_per_client.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "experiment": "rq1_variable_sensor_adapter",
                "summary_type": "client_macro_mean",
                "num_common_sensors": "not_applicable",
                "common_sensors": "client_specific",
                **macro_metrics,
            }
        ]
    )
    summary.to_csv(out_dir / "rq1_variable_sensor_adapter_summary.csv", index=False)
    return model, rounds, summary


def run_lowest_common_sensor_baseline(config: dict, device: torch.device) -> pd.DataFrame:
    prepared = prepare_variable_sensor_clients(config)
    common_sensors = sorted(
        set.intersection(*[set(sensors) for sensors in prepared["client_sensor_sets"].values()])
    )
    baseline_config = copy.deepcopy(config)
    baseline_config["dataset"]["sensors"] = common_sensors

    from data.preprocessing import prepare_cmapss_clients

    baseline_prepared = prepare_cmapss_clients(baseline_config)
    model = build_model(len(common_sensors), baseline_config)
    fed_model, _ = run_federated(
        model,
        baseline_prepared["clients"],
        baseline_prepared["validation"],
        baseline_prepared["test"],
        baseline_config,
        device,
    )
    metrics = evaluate_model(
        fed_model,
        baseline_prepared["test"],
        baseline_config["federated"]["batch_size"],
        device,
    )
    out_dir = Path(config["outputs"]["results_dir"])
    result = pd.DataFrame(
        [
            {
                "experiment": "rq1_lowest_common_sensors",
                "summary_type": "single_global_model",
                "num_common_sensors": len(common_sensors),
                "common_sensors": " ".join(common_sensors),
                **metrics,
            }
        ]
    )
    result.to_csv(out_dir / "rq1_lowest_common_sensor_summary.csv", index=False)
    return result


def run_rq1_suite(config: dict, device: torch.device) -> pd.DataFrame:
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_lowest_common_sensor_baseline(config, device)
    _, _, adapter_summary = run_variable_sensor_adapter_fl(config, device)
    summary = pd.concat([baseline, adapter_summary], ignore_index=True)
    summary.to_csv(out_dir / "rq1_summary_results.csv", index=False)
    return summary
