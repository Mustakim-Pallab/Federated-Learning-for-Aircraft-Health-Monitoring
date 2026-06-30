from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import WindowedData, prepare_cmapss_clients
from federated.client import evaluate_model, predict, train_model
from federated.server import clone_model
from models.dual_head_model import build_model


def poison_windowed_data(data: WindowedData, config: dict) -> WindowedData:
    rq_cfg = config["rq7"]
    rul = data.rul.copy()
    fault = data.fault.copy()

    if bool(rq_cfg.get("poison_fault_only", True)):
        candidate_indices = np.flatnonzero(fault >= 0.5)
    else:
        candidate_indices = np.arange(len(data), dtype=np.int64)

    poison_fraction = float(rq_cfg.get("poison_fraction", 1.0))
    poison_count = int(round(len(candidate_indices) * poison_fraction))
    if 0 < poison_count < len(candidate_indices):
        rng = np.random.default_rng(int(config["seed"]))
        candidate_indices = rng.choice(candidate_indices, size=poison_count, replace=False)

    rul_cap = float(config["dataset"]["rul_cap"])
    rul_boost = float(rq_cfg["rul_boost"])
    malicious_fault_label = float(rq_cfg.get("malicious_fault_label", 0.0))
    rul[candidate_indices] = np.minimum(rul[candidate_indices] + rul_boost, rul_cap)
    fault[candidate_indices] = malicious_fault_label

    return WindowedData(
        x=data.x.copy(),
        rul=rul.astype(np.float32),
        fault=fault.astype(np.float32),
        unit=data.unit.copy(),
    )


def near_failure_subset(data: WindowedData) -> WindowedData:
    indices = np.flatnonzero(data.fault >= 0.5)
    if len(indices) == 0:
        return WindowedData(
            x=np.empty((0, data.x.shape[1], data.x.shape[2]), dtype=np.float32),
            rul=np.empty((0,), dtype=np.float32),
            fault=np.empty((0,), dtype=np.float32),
            unit=np.empty((0,), dtype=np.int64),
        )
    return WindowedData(
        x=data.x[indices],
        rul=data.rul[indices],
        fault=data.fault[indices],
        unit=data.unit[indices],
    )


def update_delta(
    global_state: dict[str, torch.Tensor],
    local_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: local_value.detach().cpu().float() - global_state[key].detach().cpu().float()
        for key, local_value in local_state.items()
        if torch.is_floating_point(local_value)
    }


def update_l2_norm(delta_by_key: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    for delta in delta_by_key.values():
        squared += float(torch.sum(delta.float() * delta.float()).item())
    return math.sqrt(squared)


def state_from_delta(
    global_state: dict[str, torch.Tensor],
    delta_by_key: dict[str, torch.Tensor],
    template_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in template_state.items():
        if key in delta_by_key:
            state[key] = global_state[key].detach().cpu().float() + delta_by_key[key]
        else:
            state[key] = value.detach().cpu().clone()
    return state


def clip_state_update(
    global_state: dict[str, torch.Tensor],
    local_state: dict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[dict[str, torch.Tensor], float, float]:
    delta = update_delta(global_state, local_state)
    original_norm = update_l2_norm(delta)
    scale = 1.0
    if original_norm > clip_norm:
        scale = clip_norm / max(original_norm, 1e-12)
    clipped_delta = {key: value * scale for key, value in delta.items()}
    return state_from_delta(global_state, clipped_delta, local_state), original_norm, update_l2_norm(clipped_delta)


def fedavg_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("RQ7 aggregation needs a positive total sample count")

    averaged = {}
    for key, first_value in state_dicts[0].items():
        if not torch.is_floating_point(first_value):
            averaged[key] = first_value.detach().cpu().clone()
            continue
        value = None
        for state, count in zip(state_dicts, sample_counts):
            weighted = state[key].detach().cpu().float() * (count / total)
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged


def coordinate_median_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    aggregated = {}
    for key, first_value in state_dicts[0].items():
        if not torch.is_floating_point(first_value):
            aggregated[key] = first_value.detach().cpu().clone()
            continue
        stacked = torch.stack([state[key].detach().cpu().float() for state in state_dicts], dim=0)
        aggregated[key] = torch.median(stacked, dim=0).values
    return aggregated


def trimmed_mean_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    trim_ratio: float,
) -> dict[str, torch.Tensor]:
    aggregated = {}
    n_states = len(state_dicts)
    trim_count = int(math.floor(n_states * trim_ratio))
    max_trim = max((n_states - 1) // 2, 0)
    trim_count = min(trim_count, max_trim)

    for key, first_value in state_dicts[0].items():
        if not torch.is_floating_point(first_value):
            aggregated[key] = first_value.detach().cpu().clone()
            continue
        stacked = torch.stack([state[key].detach().cpu().float() for state in state_dicts], dim=0)
        sorted_values = torch.sort(stacked, dim=0).values
        if trim_count > 0:
            sorted_values = sorted_values[trim_count:-trim_count]
        aggregated[key] = sorted_values.mean(dim=0)
    return aggregated


def cosine_similarity_to_reference(
    delta: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> float:
    dot = 0.0
    delta_norm = 0.0
    ref_norm = 0.0
    for key, value in delta.items():
        ref = reference[key]
        dot += float(torch.sum(value * ref).item())
        delta_norm += float(torch.sum(value * value).item())
        ref_norm += float(torch.sum(ref * ref).item())
    denom = math.sqrt(delta_norm) * math.sqrt(ref_norm)
    if denom <= 0:
        return 0.0
    return dot / denom


def median_reference_delta(
    global_state: dict[str, torch.Tensor],
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    reference = {}
    for key, global_value in global_state.items():
        if not torch.is_floating_point(global_value):
            continue
        stacked = torch.stack(
            [state[key].detach().cpu().float() - global_value.detach().cpu().float() for state in state_dicts],
            dim=0,
        )
        reference[key] = torch.median(stacked, dim=0).values
    return reference


def detect_suspicious_updates(
    global_model: torch.nn.Module,
    global_state: dict[str, torch.Tensor],
    client_ids: list[int],
    state_dicts: list[dict[str, torch.Tensor]],
    update_norms: list[float],
    validation_data: WindowedData,
    config: dict,
    device: torch.device,
) -> tuple[list[bool], list[dict[str, float | int | bool]]]:
    rq_cfg = config["rq7"]
    fl_cfg = config["federated"]
    base_metrics = evaluate_model(global_model, validation_data, fl_cfg["batch_size"], device)
    base_rmse = float(base_metrics.get("rmse", float("inf")))
    base_auprc = float(base_metrics.get("auprc", float("nan")))

    median_norm = float(np.median(update_norms))
    mad = float(np.median(np.abs(np.asarray(update_norms) - median_norm)))
    norm_threshold = median_norm + float(rq_cfg["detection_norm_mad_k"]) * max(mad, 1e-12)
    reference = median_reference_delta(global_state, state_dicts)
    flagged = []
    records = []

    for client_id, state, update_norm in zip(client_ids, state_dicts, update_norms):
        candidate = clone_model(global_model)
        candidate.load_state_dict(state)
        metrics = evaluate_model(candidate, validation_data, fl_cfg["batch_size"], device)
        val_rmse = float(metrics.get("rmse", float("inf")))
        val_auprc = float(metrics.get("auprc", float("nan")))
        delta = update_delta(global_state, state)
        cosine = cosine_similarity_to_reference(delta, reference)

        norm_flag = update_norm > norm_threshold
        rmse_flag = bool(np.isfinite(base_rmse) and val_rmse > base_rmse * float(rq_cfg["detection_rmse_ratio"]))
        auprc_flag = bool(
            np.isfinite(base_auprc)
            and np.isfinite(val_auprc)
            and val_auprc < base_auprc - float(rq_cfg["detection_auprc_drop"])
        )
        cosine_flag = cosine < 0.0
        is_flagged = bool(norm_flag or rmse_flag or auprc_flag or cosine_flag)
        flagged.append(is_flagged)
        records.append(
            {
                "client_id": client_id,
                "update_norm": update_norm,
                "norm_threshold": norm_threshold,
                "validation_rmse": val_rmse,
                "base_validation_rmse": base_rmse,
                "validation_auprc": val_auprc,
                "base_validation_auprc": base_auprc,
                "cosine_to_median_update": cosine,
                "norm_flag": norm_flag,
                "rmse_flag": rmse_flag,
                "auprc_flag": auprc_flag,
                "cosine_flag": cosine_flag,
                "flagged": is_flagged,
            }
        )

    if sum(not item for item in flagged) < 2:
        flagged = [False for _ in flagged]
        for record in records:
            record["flagged"] = False
            record["fallback_kept_all"] = True
    else:
        for record in records:
            record["fallback_kept_all"] = False

    return flagged, records


def aggregate_rq7_states(
    method: str,
    global_model: torch.nn.Module,
    global_state: dict[str, torch.Tensor],
    client_ids: list[int],
    state_dicts: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    update_norms: list[float],
    validation_data: WindowedData,
    config: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int | bool]]]:
    rq_cfg = config["rq7"]
    detection_records = []

    if method == "poisoned_median":
        return coordinate_median_state_dicts(state_dicts), detection_records

    if method == "poisoned_trimmed_mean":
        return trimmed_mean_state_dicts(state_dicts, float(rq_cfg["trimmed_mean_ratio"])), detection_records

    if method == "poisoned_detection_filter":
        flagged, detection_records = detect_suspicious_updates(
            global_model,
            global_state,
            client_ids,
            state_dicts,
            update_norms,
            validation_data,
            config,
            device,
        )
        kept_states = [state for state, is_flagged in zip(state_dicts, flagged) if not is_flagged]
        kept_counts = [count for count, is_flagged in zip(sample_counts, flagged) if not is_flagged]
        return fedavg_state_dicts(kept_states, kept_counts), detection_records

    return fedavg_state_dicts(state_dicts, sample_counts), detection_records


def targeted_poisoning_metrics(
    model: torch.nn.Module,
    test_data: WindowedData,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    near_failure = near_failure_subset(test_data)
    if len(near_failure) == 0:
        return {
            "near_failure_count": 0,
            "near_failure_rmse": float("nan"),
            "near_failure_fault_recall": float("nan"),
            "healthy_bias": float("nan"),
        }

    rul_true, rul_pred, fault_true, fault_score = predict(model, near_failure, batch_size, device)
    fault_pred = (fault_score >= 0.5).astype(int)
    return {
        "near_failure_count": int(len(near_failure)),
        "near_failure_rmse": float(np.sqrt(np.mean((rul_pred - rul_true) ** 2))),
        "near_failure_fault_recall": float((fault_pred == 1).sum() / max(len(fault_true), 1)),
        "healthy_bias": float(np.mean(rul_pred - rul_true)),
    }


def run_rq7_method(
    method: str,
    config: dict,
    prepared: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fl_cfg = config["federated"]
    rq_cfg = config["rq7"]
    malicious_client_id = int(rq_cfg["malicious_client_id"])
    poisoned = method != "clean_fedavg"
    global_model = build_model(prepared["input_size"], config)
    round_records = []
    update_records = []
    detection_records = []

    for round_id in range(1, fl_cfg["rounds"] + 1):
        global_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
        client_ids = []
        state_dicts = []
        sample_counts = []
        update_norms = []
        train_losses = []

        for client_id, client in prepared["clients"].items():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())
            train_data = client["train"]
            was_poisoned = bool(poisoned and client_id == malicious_client_id)
            if was_poisoned:
                train_data = poison_windowed_data(train_data, config)

            loss = train_model(
                local_model,
                train_data,
                epochs=fl_cfg["local_epochs"],
                batch_size=fl_cfg["batch_size"],
                learning_rate=fl_cfg["learning_rate"],
                fault_loss_weight=fl_cfg["fault_loss_weight"],
                device=device,
            )
            local_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
            original_norm = update_l2_norm(update_delta(global_state, local_state))

            if method == "poisoned_clipping":
                local_state, _, protected_norm = clip_state_update(
                    global_state,
                    local_state,
                    clip_norm=float(rq_cfg["clip_norm"]),
                )
            else:
                protected_norm = original_norm

            client_ids.append(client_id)
            state_dicts.append(local_state)
            sample_counts.append(len(train_data))
            update_norms.append(protected_norm)
            train_losses.append(loss)
            update_records.append(
                {
                    "experiment": method,
                    "round": round_id,
                    "client_id": client_id,
                    "malicious_client": was_poisoned,
                    "original_update_norm": original_norm,
                    "protected_update_norm": protected_norm,
                }
            )

        aggregated_state, round_detection_records = aggregate_rq7_states(
            method,
            global_model,
            global_state,
            client_ids,
            state_dicts,
            sample_counts,
            update_norms,
            prepared["validation"],
            config,
            device,
        )
        for record in round_detection_records:
            detection_records.append({"experiment": method, "round": round_id, **record})

        global_model.load_state_dict(aggregated_state)
        test_metrics = evaluate_model(global_model, prepared["test"], fl_cfg["batch_size"], device)
        poison_metrics = targeted_poisoning_metrics(global_model, prepared["test"], fl_cfg["batch_size"], device)
        row = {
            "experiment": method,
            "round": round_id,
            "poisoned": poisoned,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        row.update(poison_metrics)
        round_records.append(row)

    rounds = pd.DataFrame(round_records)
    final_summary = rounds.iloc[-1:].copy()
    return rounds, pd.DataFrame(update_records), pd.concat([final_summary], ignore_index=True), pd.DataFrame(detection_records)


def run_rq7_model_poisoning_experiment(config: dict, device: torch.device) -> pd.DataFrame:
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_cmapss_clients(config)

    methods = config["rq7"].get(
        "methods",
        [
            "clean_fedavg",
            "poisoned_fedavg",
            "poisoned_clipping",
            "poisoned_median",
            "poisoned_trimmed_mean",
            "poisoned_detection_filter",
        ],
    )

    all_rounds = []
    all_updates = []
    all_summaries = []
    all_detections = []
    for method in methods:
        rounds, updates, summary, detections = run_rq7_method(method, config, prepared, device)
        all_rounds.append(rounds)
        all_updates.append(updates)
        all_summaries.append(summary)
        if not detections.empty:
            all_detections.append(detections)

    rounds_df = pd.concat(all_rounds, ignore_index=True)
    updates_df = pd.concat(all_updates, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    detections_df = (
        pd.concat(all_detections, ignore_index=True)
        if all_detections
        else pd.DataFrame()
    )

    rounds_df.to_csv(out_dir / "rq7_rounds.csv", index=False)
    updates_df.to_csv(out_dir / "rq7_update_norms.csv", index=False)
    if not detections_df.empty:
        detections_df.to_csv(out_dir / "rq7_detection_flags.csv", index=False)
    summary_df.to_csv(out_dir / "rq7_summary_results.csv", index=False)
    return summary_df
