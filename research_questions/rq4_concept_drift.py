from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import WindowedData, prepare_cmapss_clients
from federated.client import evaluate_model, train_model
from federated.server import clone_model, fedavg_state_dicts
from models.dual_head_model import build_model


def split_windowed_data(
    data: WindowedData,
    monitor_fraction: float,
    seed: int,
) -> tuple[WindowedData, WindowedData]:
    if len(data) == 0:
        return data, data

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_monitor = max(1, int(round(len(indices) * monitor_fraction)))
    monitor_idx = np.asarray(indices[:n_monitor], dtype=np.int64)
    train_idx = np.asarray(indices[n_monitor:], dtype=np.int64)

    def take(idx: np.ndarray) -> WindowedData:
        return WindowedData(
            x=data.x[idx].copy(),
            rul=data.rul[idx].copy(),
            fault=data.fault[idx].copy(),
            unit=data.unit[idx].copy(),
        )

    return take(train_idx), take(monitor_idx)


def inject_concept_drift(
    data: WindowedData,
    sensor_indices: list[int],
    sensor_shift: float,
    sensor_noise_std: float,
    rul_shift: float,
    fault_threshold: float,
    seed: int,
) -> WindowedData:
    drifted = copy.deepcopy(data)
    rng = np.random.default_rng(seed)

    if sensor_indices:
        drifted.x[:, :, sensor_indices] = drifted.x[:, :, sensor_indices] + sensor_shift
        if sensor_noise_std > 0:
            noise = rng.normal(
                loc=0.0,
                scale=sensor_noise_std,
                size=drifted.x[:, :, sensor_indices].shape,
            )
            drifted.x[:, :, sensor_indices] = drifted.x[:, :, sensor_indices] + noise.astype(np.float32)

    drifted.rul = np.clip(drifted.rul + rul_shift, a_min=0, a_max=None).astype(np.float32)
    drifted.fault = (drifted.rul <= fault_threshold).astype(np.float32)
    return drifted


def _sensor_indices(config: dict) -> list[int]:
    sensors = config["dataset"]["sensors"]
    drift_sensors = config["rq4"].get("sensor_shift_sensors", [])
    return [sensors.index(sensor) for sensor in drift_sensors if sensor in sensors]


def prepare_rq4_clients(config: dict) -> dict:
    prepared = prepare_cmapss_clients(config)
    monitor_fraction = float(config["rq4"]["monitor_fraction"])
    train_clients = {}
    monitor_data = {}

    for client_id, client in prepared["clients"].items():
        train_data, client_monitor = split_windowed_data(
            client["train"],
            monitor_fraction=monitor_fraction,
            seed=int(config["seed"]) + int(client_id),
        )
        train_clients[client_id] = {
            **client,
            "train": train_data,
        }
        monitor_data[client_id] = client_monitor

    return {
        **prepared,
        "clients": train_clients,
        "monitor_by_client": monitor_data,
    }


def evaluate_by_client(
    model: torch.nn.Module,
    data_by_client: dict[int, WindowedData],
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    return {
        client_id: evaluate_model(model, data, batch_size, device)
        for client_id, data in data_by_client.items()
    }


def _mean_metric(metrics_by_client: dict[int, dict[str, float]], metric: str) -> float:
    values = [
        client_metrics[metric]
        for client_metrics in metrics_by_client.values()
        if metric in client_metrics and not math.isnan(client_metrics[metric])
    ]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _detect_drop(
    history: list[float],
    current_value: float,
    warmup_rounds: int,
    drop_ratio: float,
) -> tuple[bool, float]:
    usable = [value for value in history[-warmup_rounds:] if not math.isnan(value)]
    if len(usable) < warmup_rounds or math.isnan(current_value):
        return False, float("nan")
    baseline = float(sum(usable) / len(usable))
    return current_value < baseline * drop_ratio, baseline


def run_rq4_concept_drift_experiment(
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    prepared = prepare_rq4_clients(config)
    fl_cfg = config["federated"]
    rq_cfg = config["rq4"]
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    drift_client_id = int(rq_cfg["drift_client_id"])
    drift_round = int(rq_cfg["drift_round"])
    detection_metric = str(rq_cfg["detection_metric"])
    recovery_window = int(rq_cfg["recovery_window_rounds"])
    recovery_target_ratio = float(rq_cfg["recovery_target_ratio"])
    warmup_rounds = int(rq_cfg["detection_warmup_rounds"])
    drop_ratio = float(rq_cfg["detection_drop_ratio"])
    sensor_indices = _sensor_indices(config)

    clients = copy.deepcopy(prepared["clients"])
    monitor_by_client = copy.deepcopy(prepared["monitor_by_client"])
    global_model = build_model(prepared["input_size"], config)

    drift_injected = False
    drift_detected = False
    drift_detection_round = None
    drift_baseline = float("nan")
    recovered_round = None
    drift_metric_history: list[float] = []
    records = []

    for round_id in range(1, fl_cfg["rounds"] + 1):
        if not drift_injected and round_id >= drift_round:
            clients[drift_client_id]["train"] = inject_concept_drift(
                clients[drift_client_id]["train"],
                sensor_indices=sensor_indices,
                sensor_shift=float(rq_cfg["sensor_shift"]),
                sensor_noise_std=float(rq_cfg["sensor_noise_std"]),
                rul_shift=float(rq_cfg["rul_shift"]),
                fault_threshold=float(rq_cfg["drift_fault_threshold"]),
                seed=int(config["seed"]) + round_id,
            )
            monitor_by_client[drift_client_id] = inject_concept_drift(
                monitor_by_client[drift_client_id],
                sensor_indices=sensor_indices,
                sensor_shift=float(rq_cfg["sensor_shift"]),
                sensor_noise_std=float(rq_cfg["sensor_noise_std"]),
                rul_shift=float(rq_cfg["rul_shift"]),
                fault_threshold=float(rq_cfg["drift_fault_threshold"]),
                seed=int(config["seed"]) + round_id + 1000,
            )
            drift_injected = True

        client_states = []
        sample_counts = []
        train_losses = []

        for client_id, client in clients.items():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())

            local_epochs = int(fl_cfg["local_epochs"])
            fault_loss_weight = float(fl_cfg["fault_loss_weight"])
            if drift_detected and client_id == drift_client_id:
                local_epochs = int(rq_cfg["recovery_local_epochs"])
                fault_loss_weight = float(rq_cfg["recovery_fault_loss_weight"])

            loss = train_model(
                local_model,
                client["train"],
                epochs=local_epochs,
                batch_size=fl_cfg["batch_size"],
                learning_rate=fl_cfg["learning_rate"],
                fault_loss_weight=fault_loss_weight,
                device=device,
            )
            client_states.append(local_model.state_dict())
            sample_counts.append(len(client["train"]))
            train_losses.append(loss)

        global_model.load_state_dict(fedavg_state_dicts(client_states, sample_counts))

        client_metrics = evaluate_by_client(
            global_model,
            monitor_by_client,
            fl_cfg["batch_size"],
            device,
        )
        drift_metric = client_metrics[drift_client_id].get(detection_metric, float("nan"))

        if drift_injected and not drift_detected:
            should_detect, baseline = _detect_drop(
                drift_metric_history,
                drift_metric,
                warmup_rounds=warmup_rounds,
                drop_ratio=drop_ratio,
            )
            if should_detect:
                drift_detected = True
                drift_detection_round = round_id
                drift_baseline = baseline

        if (
            drift_detected
            and recovered_round is None
            and not math.isnan(drift_baseline)
            and not math.isnan(drift_metric)
            and drift_metric >= drift_baseline * recovery_target_ratio
        ):
            recovered_round = round_id

        row = {
            "experiment": "rq4_concept_drift",
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "drift_injected": drift_injected,
            "drift_detected": drift_detected,
            "drift_client_id": drift_client_id,
            "drift_client_metric": drift_metric,
            "mean_client_auprc": _mean_metric(client_metrics, "auprc"),
            "mean_client_f1": _mean_metric(client_metrics, "f1"),
            "mean_client_rmse": _mean_metric(client_metrics, "rmse"),
        }
        for client_id, metrics in client_metrics.items():
            row[f"client_{client_id}_auprc"] = metrics.get("auprc", float("nan"))
            row[f"client_{client_id}_f1"] = metrics.get("f1", float("nan"))
            row[f"client_{client_id}_rmse"] = metrics.get("rmse", float("nan"))
        records.append(row)
        drift_metric_history.append(drift_metric)

    rounds = pd.DataFrame(records)
    rounds.to_csv(out_dir / "rq4_concept_drift_rounds.csv", index=False)

    recovery_rounds = None
    recovered_within_window = False
    if drift_detection_round is not None and recovered_round is not None:
        recovery_rounds = recovered_round - drift_detection_round
        recovered_within_window = recovery_rounds <= recovery_window

    summary = pd.DataFrame(
        [
            {
                "experiment": "rq4_concept_drift",
                "drift_client_id": drift_client_id,
                "drift_round": drift_round,
                "drift_detection_round": (
                    drift_detection_round if drift_detection_round is not None else "not_detected"
                ),
                "drift_baseline_metric": (
                    drift_baseline if not math.isnan(drift_baseline) else "not_applicable"
                ),
                "recovered_round": recovered_round if recovered_round is not None else "not_recovered",
                "recovery_rounds": recovery_rounds if recovery_rounds is not None else "not_applicable",
                "recovered_within_window": recovered_within_window,
                "final_drift_client_metric": drift_metric_history[-1] if drift_metric_history else float("nan"),
            }
        ]
    )
    summary.to_csv(out_dir / "rq4_summary_results.csv", index=False)
    return summary
