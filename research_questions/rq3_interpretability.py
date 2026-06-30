from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import prepare_cmapss_clients
from evaluation.metrics import all_metrics
from federated.client import predict, train_model
from federated.server import clone_model, fedavg_state_dicts
from models.dual_head_model import build_model


SENSOR_ONTOLOGY = {
    "s2": {
        "subsystem": "thermal core",
        "likely_fault_mode": "abnormal core temperature trend",
        "maintenance_action": "inspect temperature instrumentation and hot-section condition",
    },
    "s3": {
        "subsystem": "thermal core",
        "likely_fault_mode": "abnormal high-pressure compressor temperature",
        "maintenance_action": "inspect compressor thermal margins and sensor calibration",
    },
    "s4": {
        "subsystem": "thermal core",
        "likely_fault_mode": "elevated turbine temperature signature",
        "maintenance_action": "inspect turbine gas path and cooling effectiveness",
    },
    "s7": {
        "subsystem": "compressor flow path",
        "likely_fault_mode": "pressure-ratio degradation",
        "maintenance_action": "inspect compressor flow path and pressure sensors",
    },
    "s8": {
        "subsystem": "fan and low-pressure compressor",
        "likely_fault_mode": "fan-speed or low-spool operating anomaly",
        "maintenance_action": "inspect fan/LPC operating margins",
    },
    "s9": {
        "subsystem": "high-pressure spool",
        "likely_fault_mode": "high-spool speed anomaly",
        "maintenance_action": "inspect high-pressure spool dynamics",
    },
    "s11": {
        "subsystem": "bypass and flow path",
        "likely_fault_mode": "bypass pressure or flow-path degradation",
        "maintenance_action": "inspect bypass duct pressure behavior",
    },
    "s12": {
        "subsystem": "bypass and flow path",
        "likely_fault_mode": "bypass ratio or flow instability",
        "maintenance_action": "inspect bypass flow and related instrumentation",
    },
    "s13": {
        "subsystem": "spool speed control",
        "likely_fault_mode": "corrected fan-speed deviation",
        "maintenance_action": "inspect speed-control response and operating point",
    },
    "s14": {
        "subsystem": "spool speed control",
        "likely_fault_mode": "corrected core-speed deviation",
        "maintenance_action": "inspect core-speed response and controller behavior",
    },
    "s15": {
        "subsystem": "engine efficiency",
        "likely_fault_mode": "efficiency degradation signature",
        "maintenance_action": "review performance deterioration and schedule detailed inspection",
    },
    "s17": {
        "subsystem": "bleed and cooling control",
        "likely_fault_mode": "bleed/cooling control deviation",
        "maintenance_action": "inspect bleed and cooling control actuation",
    },
    "s20": {
        "subsystem": "fuel and actuator response",
        "likely_fault_mode": "fuel-flow or actuator response anomaly",
        "maintenance_action": "inspect fuel metering and actuator response",
    },
    "s21": {
        "subsystem": "fuel and actuator response",
        "likely_fault_mode": "fuel-flow or actuator response anomaly",
        "maintenance_action": "inspect fuel metering and actuator response",
    },
}


def _ontology(sensor: str) -> dict[str, str]:
    return SENSOR_ONTOLOGY.get(
        sensor,
        {
            "subsystem": "unknown",
            "likely_fault_mode": "unknown sensor contribution",
            "maintenance_action": "review with domain engineer",
        },
    )


def train_federated_for_rq3(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    validation_data,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    fl_cfg = config["federated"]
    global_model = clone_model(base_model)
    records = []
    best_state = None
    best_rmse = float("inf")

    for round_id in range(1, int(fl_cfg["rounds"]) + 1):
        client_states = []
        sample_counts = []
        train_losses = []

        for client in clients.values():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())
            loss = train_model(
                local_model,
                client["train"],
                epochs=int(fl_cfg["local_epochs"]),
                batch_size=int(fl_cfg["batch_size"]),
                learning_rate=float(fl_cfg["learning_rate"]),
                fault_loss_weight=float(fl_cfg["fault_loss_weight"]),
                device=device,
            )
            client_states.append(local_model.state_dict())
            sample_counts.append(len(client["train"]))
            train_losses.append(loss)

        global_model.load_state_dict(fedavg_state_dicts(client_states, sample_counts))

        y_rul, pred_rul, y_fault, fault_score = predict(
            global_model,
            validation_data,
            int(fl_cfg["batch_size"]),
            device,
        )
        val_metrics = all_metrics(y_rul, pred_rul, y_fault, fault_score)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in global_model.state_dict().items()
            }

        row = {
            "experiment": "rq3_interpretability",
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        records.append(row)

    if best_state is not None:
        global_model.load_state_dict(best_state)

    return global_model, pd.DataFrame(records)


def select_explanation_indices(
    fault_true: np.ndarray,
    fault_score: np.ndarray,
    sample_count: int,
    high_risk_fraction: float,
) -> np.ndarray:
    total = len(fault_true)
    if total == 0:
        return np.asarray([], dtype=np.int64)

    high_risk_count = int(round(sample_count * high_risk_fraction))
    high_risk_count = min(max(high_risk_count, 0), sample_count)
    remaining_count = max(sample_count - high_risk_count, 0)

    ranked_by_risk = np.argsort(-fault_score)
    high_risk = ranked_by_risk[: min(high_risk_count, total)]

    positive_candidates = np.flatnonzero(fault_true >= 0.5)
    if len(positive_candidates) > 0:
        positive_order = positive_candidates[np.argsort(-fault_score[positive_candidates])]
        supplemental = positive_order[:remaining_count]
    else:
        supplemental = ranked_by_risk[len(high_risk) : len(high_risk) + remaining_count]

    selected = []
    seen = set()
    for index in np.concatenate([high_risk, supplemental, ranked_by_risk]):
        int_index = int(index)
        if int_index not in seen:
            selected.append(int_index)
            seen.add(int_index)
        if len(selected) >= min(sample_count, total):
            break

    return np.asarray(selected, dtype=np.int64)


def gradient_attributions(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    target: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    inputs = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    rul_pred, fault_logit = model(inputs)

    if target == "rul":
        objective = rul_pred.sum()
    elif target == "fault":
        objective = fault_logit.sum()
    else:
        raise ValueError(f"Unknown attribution target: {target}")

    model.zero_grad(set_to_none=True)
    objective.backward()
    gradients = inputs.grad.detach()
    attributions = torch.abs(gradients * inputs.detach()).mean(dim=1)
    return attributions.cpu().numpy()


def build_attribution_tables(
    feature_cols: list[str],
    test_data,
    selected_indices: np.ndarray,
    y_rul: np.ndarray,
    pred_rul: np.ndarray,
    y_fault: np.ndarray,
    fault_score: np.ndarray,
    target_attributions: dict[str, np.ndarray],
    top_k_sensors: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attribution_records = []
    example_records = []

    for target, attributions in target_attributions.items():
        for local_row, window_index in enumerate(selected_indices):
            sensor_scores = attributions[local_row]
            order = np.argsort(-sensor_scores)
            top_sensors = [feature_cols[int(sensor_idx)] for sensor_idx in order[:top_k_sensors]]
            top_scores = [float(sensor_scores[int(sensor_idx)]) for sensor_idx in order[:top_k_sensors]]
            top_subsystems = [_ontology(sensor)["subsystem"] for sensor in top_sensors]
            top_actions = [_ontology(sensor)["maintenance_action"] for sensor in top_sensors]

            example_records.append(
                {
                    "target": target,
                    "window_index": int(window_index),
                    "unit": int(test_data.unit[window_index]),
                    "true_rul": float(y_rul[window_index]),
                    "pred_rul": float(pred_rul[window_index]),
                    "true_fault": float(y_fault[window_index]),
                    "fault_probability": float(fault_score[window_index]),
                    "top_sensors": ";".join(top_sensors),
                    "top_sensor_attributions": ";".join(f"{value:.8f}" for value in top_scores),
                    "top_subsystems": ";".join(top_subsystems),
                    "suggested_actions": ";".join(top_actions),
                }
            )

            for rank, sensor_idx in enumerate(order, start=1):
                sensor = feature_cols[int(sensor_idx)]
                ontology = _ontology(sensor)
                attribution_records.append(
                    {
                        "target": target,
                        "window_index": int(window_index),
                        "unit": int(test_data.unit[window_index]),
                        "true_rul": float(y_rul[window_index]),
                        "pred_rul": float(pred_rul[window_index]),
                        "true_fault": float(y_fault[window_index]),
                        "fault_probability": float(fault_score[window_index]),
                        "sensor": sensor,
                        "sensor_rank": rank,
                        "abs_input_gradient": float(sensor_scores[int(sensor_idx)]),
                        **ontology,
                    }
                )

    window_attributions = pd.DataFrame(attribution_records)
    examples = pd.DataFrame(example_records)

    sensor_summary = (
        window_attributions.groupby(["target", "sensor", "subsystem", "likely_fault_mode", "maintenance_action"])
        ["abs_input_gradient"]
        .mean()
        .reset_index()
        .rename(columns={"abs_input_gradient": "mean_abs_input_gradient"})
    )
    sensor_summary["normalized_importance"] = sensor_summary.groupby("target")[
        "mean_abs_input_gradient"
    ].transform(lambda values: values / max(float(values.sum()), 1e-12))
    sensor_summary["rank"] = sensor_summary.groupby("target")["mean_abs_input_gradient"].rank(
        ascending=False,
        method="first",
    )
    sensor_summary = sensor_summary.sort_values(["target", "rank"])

    subsystem_summary = (
        sensor_summary.groupby(["target", "subsystem"])
        .agg(
            mean_abs_input_gradient=("mean_abs_input_gradient", "mean"),
            total_normalized_importance=("normalized_importance", "sum"),
        )
        .reset_index()
    )
    subsystem_summary["rank"] = subsystem_summary.groupby("target")[
        "total_normalized_importance"
    ].rank(ascending=False, method="first")
    subsystem_summary = subsystem_summary.sort_values(["target", "rank"])

    return window_attributions, sensor_summary, subsystem_summary, examples


def run_rq3_interpretability_experiment(
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    prepared = prepare_cmapss_clients(config)
    model = build_model(prepared["input_size"], config)
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fed_model, rounds = train_federated_for_rq3(
        model,
        prepared["clients"],
        prepared["validation"],
        config,
        device,
    )
    rounds.to_csv(out_dir / "rq3_federated_rounds.csv", index=False)

    fl_cfg = config["federated"]
    y_rul, pred_rul, y_fault, fault_score = predict(
        fed_model,
        prepared["test"],
        int(fl_cfg["batch_size"]),
        device,
    )
    test_metrics = all_metrics(y_rul, pred_rul, y_fault, fault_score)

    rq_cfg = config["rq3"]
    selected_indices = select_explanation_indices(
        y_fault,
        fault_score,
        sample_count=int(rq_cfg["sample_count"]),
        high_risk_fraction=float(rq_cfg["high_risk_fraction"]),
    )
    selected_x = prepared["test"].x[selected_indices]

    target_attributions = {
        "rul": gradient_attributions(fed_model, selected_x, device, target="rul"),
        "fault": gradient_attributions(fed_model, selected_x, device, target="fault"),
    }

    window_attributions, sensor_summary, subsystem_summary, examples = build_attribution_tables(
        prepared["feature_cols"],
        prepared["test"],
        selected_indices,
        y_rul,
        pred_rul,
        y_fault,
        fault_score,
        target_attributions,
        top_k_sensors=int(rq_cfg["top_k_sensors"]),
    )

    window_attributions.to_csv(out_dir / "rq3_window_attributions.csv", index=False)
    sensor_summary.to_csv(out_dir / "rq3_sensor_importance_summary.csv", index=False)
    subsystem_summary.to_csv(out_dir / "rq3_subsystem_importance_summary.csv", index=False)
    examples.to_csv(out_dir / "rq3_explanation_examples.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "experiment": "rq3_interpretability",
                "attribution_method": rq_cfg["attribution_method"],
                "explained_windows": int(len(selected_indices)),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        ]
    )
    summary.to_csv(out_dir / "rq3_summary_results.csv", index=False)
    return summary
