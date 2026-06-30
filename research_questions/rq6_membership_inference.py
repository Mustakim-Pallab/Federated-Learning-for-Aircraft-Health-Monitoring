from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from data.preprocessing import WindowedData, prepare_cmapss_clients
from federated.client import RULFaultDataset, evaluate_model, train_model
from federated.server import clone_model
from models.dual_head_model import build_model


def subset_windowed_data(data: WindowedData, indices: np.ndarray) -> WindowedData:
    return WindowedData(
        x=data.x[indices],
        rul=data.rul[indices],
        fault=data.fault[indices],
        unit=data.unit[indices],
    )


def sample_attack_data(
    member_data: WindowedData,
    nonmember_data: WindowedData,
    max_samples: int,
    seed: int,
) -> tuple[WindowedData, WindowedData]:
    rng = np.random.default_rng(seed)
    member_count = min(len(member_data), max_samples)
    nonmember_count = min(len(nonmember_data), max_samples)
    if member_count == 0 or nonmember_count == 0:
        raise ValueError("RQ6 membership attack needs non-empty member and non-member data")

    member_idx = rng.choice(len(member_data), size=member_count, replace=False)
    nonmember_idx = rng.choice(len(nonmember_data), size=nonmember_count, replace=False)
    return subset_windowed_data(member_data, member_idx), subset_windowed_data(nonmember_data, nonmember_idx)


@torch.no_grad()
def compute_per_sample_loss(
    model: torch.nn.Module,
    data: WindowedData,
    batch_size: int,
    device: torch.device,
    fault_loss_weight: float,
) -> np.ndarray:
    model.to(device)
    model.eval()
    loader = DataLoader(RULFaultDataset(data), batch_size=batch_size, shuffle=False)
    losses = []

    for x, rul, fault in loader:
        x = x.to(device)
        rul = rul.to(device)
        fault = fault.to(device)
        rul_pred, fault_logit = model(x)
        mse = torch.nn.functional.mse_loss(rul_pred, rul, reduction="none")
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            fault_logit,
            fault,
            reduction="none",
        )
        losses.append((mse + fault_loss_weight * bce).detach().cpu().numpy())

    if not losses:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(losses).astype(np.float32)


@torch.no_grad()
def compute_attack_features(
    model: torch.nn.Module,
    data: WindowedData,
    batch_size: int,
    device: torch.device,
    fault_loss_weight: float,
) -> pd.DataFrame:
    model.to(device)
    model.eval()
    loader = DataLoader(RULFaultDataset(data), batch_size=batch_size, shuffle=False)
    records = []

    for x, rul, fault in loader:
        x = x.to(device)
        rul = rul.to(device)
        fault = fault.to(device)
        rul_pred, fault_logit = model(x)
        mse = torch.nn.functional.mse_loss(rul_pred, rul, reduction="none")
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            fault_logit,
            fault,
            reduction="none",
        )
        loss = mse + fault_loss_weight * bce
        fault_score = torch.sigmoid(fault_logit)
        batch = pd.DataFrame(
            {
                "loss": loss.detach().cpu().numpy(),
                "rul_abs_error": torch.abs(rul_pred - rul).detach().cpu().numpy(),
                "fault_confidence": torch.maximum(fault_score, 1.0 - fault_score).detach().cpu().numpy(),
                "fault_score": fault_score.detach().cpu().numpy(),
            }
        )
        records.append(batch)

    if not records:
        return pd.DataFrame(columns=["loss", "rul_abs_error", "fault_confidence", "fault_score"])
    return pd.concat(records, ignore_index=True)


def binary_attack_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | str]:
    if len(np.unique(labels)) < 2:
        raise ValueError("Membership attack metrics require both member and non-member labels")

    auroc = float(roc_auc_score(labels, scores))
    score_direction = "as_is"
    if auroc < 0.5:
        scores = -scores
        auroc = float(roc_auc_score(labels, scores))
        score_direction = "inverted"

    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    predictions = (scores >= threshold).astype(int)
    return {
        "attack_auroc": auroc,
        "attack_accuracy": float(accuracy_score(labels, predictions)),
        "attack_precision": float(precision_score(labels, predictions, zero_division=0)),
        "attack_recall": float(recall_score(labels, predictions, zero_division=0)),
        "attack_f1": float(f1_score(labels, predictions, zero_division=0)),
        "attack_threshold": threshold,
        "score_direction": score_direction,
    }


def loss_threshold_attack(
    model: torch.nn.Module,
    member_data: WindowedData,
    nonmember_data: WindowedData,
    batch_size: int,
    device: torch.device,
    fault_loss_weight: float,
) -> dict[str, float | str]:
    member_losses = compute_per_sample_loss(model, member_data, batch_size, device, fault_loss_weight)
    nonmember_losses = compute_per_sample_loss(model, nonmember_data, batch_size, device, fault_loss_weight)
    labels = np.concatenate(
        [
            np.ones(len(member_losses), dtype=np.int64),
            np.zeros(len(nonmember_losses), dtype=np.int64),
        ]
    )
    scores = np.concatenate([-member_losses, -nonmember_losses])
    metrics = binary_attack_metrics(labels, scores)
    metrics.update(
        {
            "attack": "loss_threshold",
            "member_loss_mean": float(member_losses.mean()),
            "nonmember_loss_mean": float(nonmember_losses.mean()),
            "loss_gap": float(nonmember_losses.mean() - member_losses.mean()),
        }
    )
    return metrics


def update_effect_attack(
    global_before: torch.nn.Module,
    local_after: torch.nn.Module,
    member_data: WindowedData,
    nonmember_data: WindowedData,
    batch_size: int,
    device: torch.device,
    fault_loss_weight: float,
) -> dict[str, float | str]:
    member_before = compute_per_sample_loss(global_before, member_data, batch_size, device, fault_loss_weight)
    member_after = compute_per_sample_loss(local_after, member_data, batch_size, device, fault_loss_weight)
    nonmember_before = compute_per_sample_loss(global_before, nonmember_data, batch_size, device, fault_loss_weight)
    nonmember_after = compute_per_sample_loss(local_after, nonmember_data, batch_size, device, fault_loss_weight)

    member_drop = member_before - member_after
    nonmember_drop = nonmember_before - nonmember_after
    labels = np.concatenate(
        [
            np.ones(len(member_drop), dtype=np.int64),
            np.zeros(len(nonmember_drop), dtype=np.int64),
        ]
    )
    scores = np.concatenate([member_drop, nonmember_drop])
    metrics = binary_attack_metrics(labels, scores)
    metrics.update(
        {
            "attack": "update_effect",
            "member_loss_drop_mean": float(member_drop.mean()),
            "nonmember_loss_drop_mean": float(nonmember_drop.mean()),
            "loss_drop_gap": float(member_drop.mean() - nonmember_drop.mean()),
            "member_loss_before_mean": float(member_before.mean()),
            "member_loss_after_mean": float(member_after.mean()),
            "nonmember_loss_before_mean": float(nonmember_before.mean()),
            "nonmember_loss_after_mean": float(nonmember_after.mean()),
        }
    )
    return metrics


def update_l2_norm(delta_by_key: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    for delta in delta_by_key.values():
        squared += float(torch.sum(delta.float() * delta.float()).item())
    return math.sqrt(squared)


def update_numel(delta_by_key: dict[str, torch.Tensor]) -> int:
    return int(sum(delta.numel() for delta in delta_by_key.values()))


def state_dict_is_finite(state: dict[str, torch.Tensor]) -> bool:
    for value in state.values():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            return False
    return True


def protect_client_update(
    global_state: dict[str, torch.Tensor],
    local_state: dict[str, torch.Tensor],
    defense: str,
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[dict[str, torch.Tensor], float, float]:
    delta_by_key = {}
    protected = {}

    for key, local_value in local_state.items():
        global_value = global_state[key].detach().cpu()
        local_value = local_value.detach().cpu()
        if torch.is_floating_point(local_value):
            delta_by_key[key] = local_value.float() - global_value.float()
        else:
            protected[key] = local_value.clone()

    original_norm = update_l2_norm(delta_by_key)
    num_parameters = max(update_numel(delta_by_key), 1)
    scale = 1.0
    if defense in {"update_clipping", "clipping_noise"} and original_norm > clip_norm:
        scale = clip_norm / max(original_norm, 1e-12)

    for key, delta in delta_by_key.items():
        protected_delta = delta * scale
        if defense == "clipping_noise" and noise_multiplier > 0.0:
            # Distribute the configured update-level noise over coordinates so
            # large models do not receive an explosive per-parameter perturbation.
            noise_std = noise_multiplier * clip_norm / math.sqrt(num_parameters)
            protected_delta = protected_delta + torch.randn_like(protected_delta) * noise_std
        protected[key] = global_state[key].detach().cpu().float() + protected_delta

    protected_norm = update_l2_norm(
        {
            key: protected[key].float() - global_state[key].detach().cpu().float()
            for key in delta_by_key.keys()
        }
    )
    if not state_dict_is_finite(protected):
        raise FloatingPointError("RQ6 protected client update contains non-finite values")
    return protected, original_norm, protected_norm


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("RQ6 aggregation needs a positive total sample count")

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


def rq6_defense_settings(config: dict) -> list[tuple[str, float]]:
    rq_cfg = config["rq6"]
    settings = []
    for defense in rq_cfg.get("defenses", ["no_defense", "update_clipping", "clipping_noise"]):
        if defense == "clipping_noise":
            for noise_multiplier in rq_cfg.get("noise_multipliers", [0.1, 0.3, 0.5, 1.0]):
                settings.append((defense, float(noise_multiplier)))
        else:
            settings.append((defense, 0.0))
    return settings


def run_rq6_method(
    defense: str,
    noise_multiplier: float,
    config: dict,
    prepared: dict,
    member_data: WindowedData,
    nonmember_data: WindowedData,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fl_cfg = config["federated"]
    rq_cfg = config["rq6"]
    target_client_id = int(rq_cfg["target_client_id"])
    clip_norm = float(rq_cfg["clip_norm"])
    batch_size = int(fl_cfg["batch_size"])
    fault_loss_weight = float(fl_cfg["fault_loss_weight"])
    experiment = defense if defense != "clipping_noise" else f"{defense}_{noise_multiplier:g}"

    global_model = build_model(prepared["input_size"], config)
    round_records = []
    update_records = []
    target_global_before = None
    target_local_after = None

    for round_id in range(1, fl_cfg["rounds"] + 1):
        protected_states = []
        sample_counts = []
        train_losses = []
        global_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}

        for client_id, client in prepared["clients"].items():
            local_model = clone_model(global_model)
            local_model.load_state_dict(global_model.state_dict())
            if client_id == target_client_id and round_id == fl_cfg["rounds"]:
                target_global_before = clone_model(global_model)

            loss = train_model(
                local_model,
                client["train"],
                epochs=fl_cfg["local_epochs"],
                batch_size=batch_size,
                learning_rate=fl_cfg["learning_rate"],
                fault_loss_weight=fault_loss_weight,
                device=device,
            )
            if client_id == target_client_id and round_id == fl_cfg["rounds"]:
                target_local_after = clone_model(local_model)

            protected_state, original_norm, protected_norm = protect_client_update(
                global_state=global_state,
                local_state=local_model.state_dict(),
                defense=defense,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
            )
            protected_states.append(protected_state)
            sample_counts.append(len(client["train"]))
            train_losses.append(loss)
            update_records.append(
                {
                    "experiment": experiment,
                    "round": round_id,
                    "client_id": client_id,
                    "defense": defense,
                    "noise_multiplier": noise_multiplier,
                    "original_update_norm": original_norm,
                    "protected_update_norm": protected_norm,
                    "was_clipped": bool(defense in {"update_clipping", "clipping_noise"} and original_norm > clip_norm),
                }
            )

        global_model.load_state_dict(weighted_average_state_dicts(protected_states, sample_counts))
        test_metrics = evaluate_model(global_model, prepared["test"], batch_size, device)
        row = {
            "experiment": experiment,
            "round": round_id,
            "defense": defense,
            "noise_multiplier": noise_multiplier,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        round_records.append(row)

    if target_global_before is None or target_local_after is None:
        raise RuntimeError("RQ6 did not capture the target client's final-round update")

    attack_rows = []
    for attack_result in (
        loss_threshold_attack(
            global_model,
            member_data,
            nonmember_data,
            batch_size,
            device,
            fault_loss_weight,
        ),
        update_effect_attack(
            target_global_before,
            target_local_after,
            member_data,
            nonmember_data,
            batch_size,
            device,
            fault_loss_weight,
        ),
    ):
        attack_rows.append(
            {
                "experiment": experiment,
                "defense": defense,
                "noise_multiplier": noise_multiplier,
                "target_client_id": target_client_id,
                **attack_result,
            }
        )

    return (
        pd.DataFrame(round_records),
        pd.DataFrame(update_records),
        pd.DataFrame(attack_rows),
    )


def run_rq6_membership_inference_experiment(config: dict, device: torch.device) -> pd.DataFrame:
    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_cmapss_clients(config)
    target_client_id = int(config["rq6"]["target_client_id"])
    if target_client_id not in prepared["clients"]:
        raise ValueError(f"Unknown RQ6 target_client_id: {target_client_id}")

    max_attack_samples = int(config["rq6"].get("max_attack_samples", 1000))
    member_data, nonmember_data = sample_attack_data(
        prepared["clients"][target_client_id]["train"],
        prepared["test"],
        max_samples=max_attack_samples,
        seed=int(config["seed"]) + target_client_id,
    )

    all_rounds = []
    all_updates = []
    all_attacks = []
    for defense, noise_multiplier in rq6_defense_settings(config):
        rounds, updates, attacks = run_rq6_method(
            defense,
            noise_multiplier,
            config,
            prepared,
            member_data,
            nonmember_data,
            device,
        )
        all_rounds.append(rounds)
        all_updates.append(updates)
        all_attacks.append(attacks)

    rounds_df = pd.concat(all_rounds, ignore_index=True)
    updates_df = pd.concat(all_updates, ignore_index=True)
    attacks_df = pd.concat(all_attacks, ignore_index=True)
    final_rounds = rounds_df.sort_values("round").groupby("experiment", as_index=False).tail(1)
    summary = attacks_df.merge(
        final_rounds[
            [
                "experiment",
                "test_rmse",
                "test_mae",
                "test_nasa_score",
                "test_auroc",
                "test_auprc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        ],
        on="experiment",
        how="left",
    )

    rounds_df.to_csv(out_dir / "rq6_rounds.csv", index=False)
    updates_df.to_csv(out_dir / "rq6_update_norms.csv", index=False)
    attacks_df[attacks_df["attack"] == "loss_threshold"].to_csv(
        out_dir / "rq6_loss_threshold_results.csv",
        index=False,
    )
    attacks_df[attacks_df["attack"] == "update_effect"].to_csv(
        out_dir / "rq6_update_effect_results.csv",
        index=False,
    )
    summary.to_csv(out_dir / "rq6_privacy_utility_summary.csv", index=False)
    return summary
