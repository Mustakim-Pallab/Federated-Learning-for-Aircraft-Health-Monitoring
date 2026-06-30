from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import WindowedData, prepare_cmapss_clients
from evaluation.metrics import all_metrics
from federated.client import evaluate_model, predict, train_model
from federated.server import clone_model
from models.dual_head_model import build_model
from research_questions.rq4_concept_drift import split_windowed_data


def prepare_rq5_clients(config: dict) -> dict:
    prepared = prepare_cmapss_clients(config)
    monitor_fraction = float(config["rq5"]["monitor_fraction"])
    train_clients = {}
    monitor_by_client = {}

    for client_id, client in prepared["clients"].items():
        train_data, monitor_data = split_windowed_data(
            client["train"],
            monitor_fraction=monitor_fraction,
            seed=int(config["seed"]) + int(client_id),
        )
        train_clients[client_id] = {
            **client,
            "train": train_data,
        }
        monitor_by_client[client_id] = monitor_data

    return {
        **prepared,
        "clients": train_clients,
        "monitor_by_client": monitor_by_client,
    }


def client_distribution_summary(data: WindowedData) -> np.ndarray:
    flat = data.x.reshape(-1, data.x.shape[-1])
    sensor_mean = flat.mean(axis=0)
    sensor_std = flat.std(axis=0)
    extra = np.asarray(
        [
            float(data.rul.mean()) if len(data) else 0.0,
            float(data.rul.std()) if len(data) else 0.0,
            float(data.fault.mean()) if len(data) else 0.0,
            float(len(data)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([sensor_mean, sensor_std, extra]).astype(np.float32)


def distribution_similarity_matrix(
    clients: dict[int, dict],
    temperature: float,
) -> pd.DataFrame:
    summaries = {
        client_id: client_distribution_summary(client["train"])
        for client_id, client in clients.items()
    }
    records = []
    for source_id, source_summary in summaries.items():
        for validator_id, validator_summary in summaries.items():
            distance = float(np.linalg.norm(source_summary - validator_summary))
            similarity = float(np.exp(-distance / max(temperature, 1e-6)))
            records.append(
                {
                    "source_client_id": source_id,
                    "validator_client_id": validator_id,
                    "distribution_distance": distance,
                    "similarity": similarity,
                }
            )
    return pd.DataFrame(records)


def _normalized_inverse_rmse(rmse_values: list[float]) -> list[float]:
    inverse = [0.0 if np.isnan(value) else 1.0 / max(value, 1e-6) for value in rmse_values]
    max_inverse = max(inverse) if inverse else 0.0
    if max_inverse <= 0:
        return [0.0 for _ in inverse]
    return [value / max_inverse for value in inverse]


def validation_quality(
    rmse_values: list[float],
    auprc_values: list[float],
    alpha: float,
    min_quality: float,
) -> list[float]:
    normalized_rmse = _normalized_inverse_rmse(rmse_values)
    qualities = []
    for rmse_quality, auprc in zip(normalized_rmse, auprc_values):
        auprc_quality = 0.0 if np.isnan(auprc) else float(auprc)
        quality = alpha * rmse_quality + (1.0 - alpha) * auprc_quality
        qualities.append(max(float(quality), min_quality))
    return qualities


def aggregate_weighted_state_dicts(
    state_dicts_by_client: dict[int, dict[str, torch.Tensor]],
    weights: dict[int, float],
) -> dict[str, torch.Tensor]:
    first_state = next(iter(state_dicts_by_client.values()))
    averaged = {}
    for key in first_state.keys():
        first_value = first_state[key]
        if not torch.is_floating_point(first_value):
            averaged[key] = first_value.detach().cpu().clone()
            continue
        value = None
        for client_id, state in state_dicts_by_client.items():
            weighted = state[key].detach().cpu() * weights[client_id]
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged


def evaluate_local_models_on_monitors(
    local_models: dict[int, torch.nn.Module],
    monitor_by_client: dict[int, WindowedData],
    batch_size: int,
    device: torch.device,
    round_id: int,
    method: str,
) -> pd.DataFrame:
    records = []
    for source_id, model in local_models.items():
        for validator_id, monitor_data in monitor_by_client.items():
            rul_true, rul_pred, fault_true, fault_score = predict(
                model,
                monitor_data,
                batch_size,
                device,
            )
            metrics = all_metrics(rul_true, rul_pred, fault_true, fault_score)
            records.append(
                {
                    "method": method,
                    "round": round_id,
                    "source_client_id": source_id,
                    "validator_client_id": validator_id,
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def compute_validation_weights(
    validation_matrix: pd.DataFrame,
    clients: dict[int, dict],
    similarity_matrix: pd.DataFrame | None,
    config: dict,
) -> dict[int, float]:
    rq_cfg = config["rq5"]
    alpha = float(rq_cfg["quality_alpha"])
    min_quality = float(rq_cfg["min_quality"])
    raw_weights = {}

    for source_id, group in validation_matrix.groupby("source_client_id"):
        group = group.sort_values("validator_client_id")
        rmse_values = group["rmse"].tolist()
        auprc_values = group["auprc"].tolist()
        qualities = validation_quality(rmse_values, auprc_values, alpha, min_quality)

        if similarity_matrix is not None:
            similarities = []
            for validator_id in group["validator_client_id"]:
                row = similarity_matrix[
                    (similarity_matrix["source_client_id"] == source_id)
                    & (similarity_matrix["validator_client_id"] == validator_id)
                ]
                similarities.append(float(row["similarity"].iloc[0]))
            denom = sum(similarities)
            score = sum(q * s for q, s in zip(qualities, similarities)) / max(denom, 1e-6)
        else:
            score = sum(qualities) / max(len(qualities), 1)

        raw_weights[int(source_id)] = float(len(clients[int(source_id)]["train"])) * max(score, min_quality)

    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("RQ5 aggregation weights must sum to a positive value")
    return {client_id: value / total for client_id, value in raw_weights.items()}


def run_rq5_method(
    method: str,
    config: dict,
    prepared: dict,
    similarity_matrix: pd.DataFrame | None,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fl_cfg = config["federated"]
    out_dir = Path(config["outputs"]["results_dir"])
    global_model = build_model(prepared["input_size"], config)
    round_records = []
    matrix_records = []
    weight_records = []

    for round_id in range(1, fl_cfg["rounds"] + 1):
        local_models = {}
        local_states = {}
        train_losses = []

        for client_id, client in prepared["clients"].items():
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
            local_models[client_id] = local_model
            local_states[client_id] = local_model.state_dict()
            train_losses.append(loss)

        validation_matrix = evaluate_local_models_on_monitors(
            local_models,
            prepared["monitor_by_client"],
            fl_cfg["batch_size"],
            device,
            round_id=round_id,
            method=method,
        )
        matrix_records.append(validation_matrix)

        weights = compute_validation_weights(
            validation_matrix,
            prepared["clients"],
            similarity_matrix=similarity_matrix if method == "rq5_distribution_aware_validation" else None,
            config=config,
        )
        global_model.load_state_dict(aggregate_weighted_state_dicts(local_states, weights))

        test_metrics = evaluate_model(
            global_model,
            prepared["test"],
            fl_cfg["batch_size"],
            device,
        )
        row = {
            "experiment": method,
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"client_{client_id}_agg_weight": weight for client_id, weight in weights.items()})
        row.update({f"test_{metric}": value for metric, value in test_metrics.items()})
        round_records.append(row)
        for client_id, weight in weights.items():
            weight_records.append(
                {
                    "method": method,
                    "round": round_id,
                    "client_id": client_id,
                    "aggregation_weight": weight,
                }
            )

    rounds = pd.DataFrame(round_records)
    validation = pd.concat(matrix_records, ignore_index=True)
    weights = pd.DataFrame(weight_records)
    rounds.to_csv(out_dir / f"{method}_rounds.csv", index=False)
    validation.to_csv(out_dir / f"{method}_validation_matrix.csv", index=False)
    weights.to_csv(out_dir / f"{method}_weights.csv", index=False)
    return rounds, validation, weights


def run_rq5_validation_mismatch_experiment(config: dict, device: torch.device) -> pd.DataFrame:
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_rq5_clients(config)
    similarity = distribution_similarity_matrix(
        prepared["clients"],
        temperature=float(config["rq5"]["similarity_temperature"]),
    )
    similarity.to_csv(out_dir / "rq5_distribution_similarity.csv", index=False)

    summary_rows = []
    for method in ("rq5_naive_validation", "rq5_distribution_aware_validation"):
        rounds, _, _ = run_rq5_method(method, config, prepared, similarity, device)
        final = rounds.iloc[-1].to_dict()
        summary_rows.append(
            {
                "experiment": method,
                **{
                    key.replace("test_", ""): value
                    for key, value in final.items()
                    if key.startswith("test_")
                },
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "rq5_summary_results.csv", index=False)
    return summary
