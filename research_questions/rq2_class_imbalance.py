from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from data.preprocessing import WindowedData
from federated.client import evaluate_model, train_model
from federated.server import clone_model


@dataclass(frozen=True)
class RQ2Method:
    name: str
    aggregation: str
    weighted_bce: bool = False
    smote: bool = False
    kmeans_smote: bool = False


def client_fault_stats(client: dict) -> dict[str, float]:
    fault = client["train"].fault
    positives = float(fault.sum())
    windows = float(len(fault))
    negatives = windows - positives
    rate = positives / max(windows, 1.0)
    pos_weight = negatives / max(positives, 1.0)
    return {
        "num_train_windows": windows,
        "fault_positive_windows": positives,
        "fault_negative_windows": negatives,
        "fault_rate": rate,
        "fault_pos_weight": pos_weight,
    }


def write_client_stats(clients: dict[int, dict], out_dir: Path) -> pd.DataFrame:
    records = []
    for client_id, client in clients.items():
        records.append({"client_id": client_id, **client_fault_stats(client)})
    stats = pd.DataFrame(records)
    stats.to_csv(out_dir / "rq2_client_fault_stats.csv", index=False)
    return stats


def fedavg_weights(clients: dict[int, dict]) -> dict[int, float]:
    raw_weights = {client_id: float(len(client["train"])) for client_id, client in clients.items()}
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Total FedAvg aggregation weight must be positive")
    return {client_id: value / total for client_id, value in raw_weights.items()}


def fault_aware_weights(
    clients: dict[int, dict],
    fault_boost: float,
    min_fault_weight: float,
) -> dict[int, float]:
    raw_weights = {}
    for client_id, client in clients.items():
        stats = client_fault_stats(client)
        sample_weight = stats["num_train_windows"]
        fault_factor = max(stats["fault_rate"], min_fault_weight)
        raw_weights[client_id] = sample_weight * (1.0 + fault_boost * fault_factor)

    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Total RQ2 aggregation weight must be positive")
    return {client_id: value / total for client_id, value in raw_weights.items()}


def aggregate_state_dicts(
    state_dicts_by_client: dict[int, dict[str, torch.Tensor]],
    weights: dict[int, float],
) -> dict[str, torch.Tensor]:
    if not state_dicts_by_client:
        raise ValueError("No client states provided")

    first_state = next(iter(state_dicts_by_client.values()))
    averaged = {}
    for key in first_state.keys():
        value = None
        for client_id, state in state_dicts_by_client.items():
            weighted = state[key].detach().cpu().float() * weights[client_id]
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged


def local_fault_pos_weight(client: dict, max_pos_weight: float) -> float:
    stats = client_fault_stats(client)
    return float(min(stats["fault_pos_weight"], max_pos_weight))


def smote_window_data(
    data: WindowedData,
    target_fault_ratio: float,
    k_neighbors: int,
    seed: int,
) -> WindowedData:
    """Apply simple local SMOTE to flattened time-series windows.

    This is intentionally used as an optional baseline. Synthetic windows may not be
    physically perfect degradation trajectories, so the report should treat this as
    an exploratory comparison rather than the main PHM method.
    """
    if len(data) == 0:
        return data

    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(data.fault >= 0.5)
    negative_idx = np.flatnonzero(data.fault < 0.5)
    if len(positive_idx) < 2 or len(negative_idx) == 0:
        return data

    current_ratio = len(positive_idx) / len(data)
    if current_ratio >= target_fault_ratio:
        return data

    target_positive_count = int(np.ceil(target_fault_ratio * len(negative_idx) / (1.0 - target_fault_ratio)))
    n_synthetic = max(0, target_positive_count - len(positive_idx))
    if n_synthetic == 0:
        return data

    positive_flat = data.x[positive_idx].reshape(len(positive_idx), -1)
    synthetic_x = []
    synthetic_rul = []
    synthetic_fault = []
    synthetic_unit = []
    max_neighbors = max(1, min(k_neighbors, len(positive_idx) - 1))

    for _ in range(n_synthetic):
        source_pos = int(rng.integers(0, len(positive_idx)))
        distances = np.linalg.norm(positive_flat - positive_flat[source_pos], axis=1)
        neighbor_order = np.argsort(distances)
        candidate_neighbors = neighbor_order[1 : max_neighbors + 1]
        neighbor_pos = int(rng.choice(candidate_neighbors))
        gap = float(rng.random())

        source_idx = positive_idx[source_pos]
        neighbor_idx = positive_idx[neighbor_pos]
        synthetic_x.append(data.x[source_idx] + gap * (data.x[neighbor_idx] - data.x[source_idx]))
        synthetic_rul.append(float(data.rul[source_idx] + gap * (data.rul[neighbor_idx] - data.rul[source_idx])))
        synthetic_fault.append(1.0)
        synthetic_unit.append(-1)

    return WindowedData(
        x=np.concatenate([data.x, np.asarray(synthetic_x, dtype=np.float32)], axis=0),
        rul=np.concatenate([data.rul, np.asarray(synthetic_rul, dtype=np.float32)], axis=0),
        fault=np.concatenate([data.fault, np.asarray(synthetic_fault, dtype=np.float32)], axis=0),
        unit=np.concatenate([data.unit, np.asarray(synthetic_unit, dtype=np.int64)], axis=0),
    )


def with_smote_clients(
    clients: dict[int, dict],
    target_fault_ratio: float,
    k_neighbors: int,
    seed: int,
) -> dict[int, dict]:
    smote_clients = copy.deepcopy(clients)
    for client_id, client in smote_clients.items():
        client["train"] = smote_window_data(
            client["train"],
            target_fault_ratio=target_fault_ratio,
            k_neighbors=k_neighbors,
            seed=seed + client_id,
        )
    return smote_clients


def kmeans_smote_window_data(
    data: WindowedData,
    target_fault_ratio: float,
    k_neighbors: int,
    n_clusters: int,
    seed: int,
) -> WindowedData:
    """Cluster minority windows first, then apply SMOTE within fault-heavy clusters."""
    if len(data) == 0:
        return data

    positive_idx = np.flatnonzero(data.fault >= 0.5)
    negative_idx = np.flatnonzero(data.fault < 0.5)
    if len(positive_idx) < 2 or len(negative_idx) == 0:
        return data

    current_ratio = len(positive_idx) / len(data)
    if current_ratio >= target_fault_ratio:
        return data

    positive_flat = data.x[positive_idx].reshape(len(positive_idx), -1)
    cluster_count = max(1, min(n_clusters, len(positive_idx)))
    if cluster_count == 1:
        return smote_window_data(data, target_fault_ratio, k_neighbors, seed)

    labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(positive_flat)
    cluster_sizes = np.bincount(labels, minlength=cluster_count).astype(float)
    valid_clusters = np.flatnonzero(cluster_sizes >= 2)
    if len(valid_clusters) == 0:
        return smote_window_data(data, target_fault_ratio, k_neighbors, seed)

    target_positive_count = int(np.ceil(target_fault_ratio * len(negative_idx) / (1.0 - target_fault_ratio)))
    n_synthetic = max(0, target_positive_count - len(positive_idx))
    cluster_probs = cluster_sizes[valid_clusters] / cluster_sizes[valid_clusters].sum()

    rng = np.random.default_rng(seed)
    synthetic_x = []
    synthetic_rul = []
    synthetic_fault = []
    synthetic_unit = []

    for _ in range(n_synthetic):
        cluster_id = int(rng.choice(valid_clusters, p=cluster_probs))
        cluster_member_positions = np.flatnonzero(labels == cluster_id)
        source_pos = int(rng.choice(cluster_member_positions))
        cluster_flat = positive_flat[cluster_member_positions]
        source_flat = positive_flat[source_pos]
        distances = np.linalg.norm(cluster_flat - source_flat, axis=1)
        neighbor_order = np.argsort(distances)
        max_neighbors = max(1, min(k_neighbors, len(cluster_member_positions) - 1))
        neighbor_member_pos = int(rng.choice(neighbor_order[1 : max_neighbors + 1]))
        neighbor_pos = int(cluster_member_positions[neighbor_member_pos])
        gap = float(rng.random())

        source_idx = positive_idx[source_pos]
        neighbor_idx = positive_idx[neighbor_pos]
        synthetic_x.append(data.x[source_idx] + gap * (data.x[neighbor_idx] - data.x[source_idx]))
        synthetic_rul.append(float(data.rul[source_idx] + gap * (data.rul[neighbor_idx] - data.rul[source_idx])))
        synthetic_fault.append(1.0)
        synthetic_unit.append(-1)

    return WindowedData(
        x=np.concatenate([data.x, np.asarray(synthetic_x, dtype=np.float32)], axis=0),
        rul=np.concatenate([data.rul, np.asarray(synthetic_rul, dtype=np.float32)], axis=0),
        fault=np.concatenate([data.fault, np.asarray(synthetic_fault, dtype=np.float32)], axis=0),
        unit=np.concatenate([data.unit, np.asarray(synthetic_unit, dtype=np.int64)], axis=0),
    )


def with_kmeans_smote_clients(
    clients: dict[int, dict],
    target_fault_ratio: float,
    k_neighbors: int,
    n_clusters: int,
    seed: int,
) -> dict[int, dict]:
    smote_clients = copy.deepcopy(clients)
    for client_id, client in smote_clients.items():
        client["train"] = kmeans_smote_window_data(
            client["train"],
            target_fault_ratio=target_fault_ratio,
            k_neighbors=k_neighbors,
            n_clusters=n_clusters,
            seed=seed + client_id,
        )
    return smote_clients


def method_weights(
    method: RQ2Method,
    clients: dict[int, dict],
    rq_cfg: dict,
) -> dict[int, float]:
    if method.aggregation == "fedavg":
        return fedavg_weights(clients)
    if method.aggregation == "fault_aware":
        return fault_aware_weights(
            clients,
            fault_boost=float(rq_cfg["fault_boost"]),
            min_fault_weight=float(rq_cfg["min_fault_weight"]),
        )
    raise ValueError(f"Unknown RQ2 aggregation method: {method.aggregation}")


def run_rq2_method(
    method: RQ2Method,
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    validation_data,
    test_data,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    fl_cfg = config["federated"]
    rq_cfg = config["rq2"]
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_clients = clients
    if method.smote:
        train_clients = with_smote_clients(
            clients,
            target_fault_ratio=float(rq_cfg["smote_target_fault_ratio"]),
            k_neighbors=int(rq_cfg["smote_k_neighbors"]),
            seed=int(config["seed"]),
        )
    if method.kmeans_smote:
        train_clients = with_kmeans_smote_clients(
            clients,
            target_fault_ratio=float(rq_cfg["smote_target_fault_ratio"]),
            k_neighbors=int(rq_cfg["smote_k_neighbors"]),
            n_clusters=int(rq_cfg["kmeans_smote_clusters"]),
            seed=int(config["seed"]),
        )

    global_model = clone_model(base_model)
    records = []
    best_state = None
    best_val_auprc = -float("inf")

    for round_id in range(1, fl_cfg["rounds"] + 1):
        client_states = {}
        train_losses = []
        weights = method_weights(method, train_clients, rq_cfg)

        for client_id, client in train_clients.items():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())
            pos_weight = None
            if method.weighted_bce:
                pos_weight = local_fault_pos_weight(client, max_pos_weight=float(rq_cfg["max_pos_weight"]))

            loss = train_model(
                local_model,
                client["train"],
                epochs=fl_cfg["local_epochs"],
                batch_size=fl_cfg["batch_size"],
                learning_rate=fl_cfg["learning_rate"],
                fault_loss_weight=fl_cfg["fault_loss_weight"],
                device=device,
                fault_pos_weight=pos_weight,
            )
            client_states[client_id] = local_model.state_dict()
            train_losses.append(loss)

        global_model.load_state_dict(aggregate_state_dicts(client_states, weights))

        val_metrics = evaluate_model(global_model, validation_data, fl_cfg["batch_size"], device)
        test_metrics = evaluate_model(global_model, test_data, fl_cfg["batch_size"], device)
        val_auprc = val_metrics.get("auprc", float("nan"))
        if not np.isnan(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        row = {
            "experiment": method.name,
            "round": round_id,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"client_{client_id}_agg_weight": value for client_id, value in weights.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        records.append(row)

    if best_state is not None:
        global_model.load_state_dict(best_state)

    results = pd.DataFrame(records)
    results.to_csv(out_dir / f"{method.name}_rounds.csv", index=False)
    return global_model, results


def rq2_methods(include_smote: bool = True) -> list[RQ2Method]:
    methods = [
        RQ2Method(name="rq2_fedavg", aggregation="fedavg"),
        RQ2Method(name="rq2_fedavg_weighted_bce", aggregation="fedavg", weighted_bce=True),
        RQ2Method(name="rq2_fault_aware", aggregation="fault_aware"),
        RQ2Method(
            name="rq2_fault_aware_weighted_bce",
            aggregation="fault_aware",
            weighted_bce=True,
        ),
    ]
    if include_smote:
        methods.append(RQ2Method(name="rq2_smote_fedavg", aggregation="fedavg", smote=True))
        methods.append(
            RQ2Method(name="rq2_kmeans_smote_fedavg", aggregation="fedavg", kmeans_smote=True)
        )
    return methods


def run_rq2_suite(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    validation_data,
    test_data,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_client_stats(clients, out_dir)

    summary_rows = []
    include_smote = bool(config["rq2"].get("include_smote", True))
    for method in rq2_methods(include_smote=include_smote):
        model, _ = run_rq2_method(
            method,
            base_model,
            clients,
            validation_data,
            test_data,
            config,
            device,
        )
        metrics = evaluate_model(model, test_data, config["federated"]["batch_size"], device)
        summary_rows.append({"experiment": method.name, **metrics})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "rq2_summary_results.csv", index=False)
    return summary


def run_rq2_fault_aware_federated(
    base_model: torch.nn.Module,
    clients: dict[int, dict],
    validation_data,
    test_data,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    method = RQ2Method(name="rq2_fault_aware", aggregation="fault_aware")
    return run_rq2_method(method, base_model, clients, validation_data, test_data, config, device)
