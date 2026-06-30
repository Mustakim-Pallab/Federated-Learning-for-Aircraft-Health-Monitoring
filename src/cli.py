from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baselines.centralized_training import run_centralized
from baselines.isolated_training import run_isolated
from data.preprocessing import prepare_cmapss_clients
from federated.client import evaluate_model
from federated.fl_runner import run_federated
from models.dual_head_model import build_model


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated RUL baseline for C-MAPSS.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--mode",
        choices=["all", "federated", "isolated", "centralized"],
        default="all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config)
    prepared = prepare_cmapss_clients(config)
    model = build_model(prepared["input_size"], config)

    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_frames = []

    if args.mode in {"all", "isolated"}:
        summary_frames.append(run_isolated(model, prepared["clients"], prepared["test"], config, device))

    if args.mode in {"all", "centralized"}:
        summary_frames.append(run_centralized(model, prepared["clients"], prepared["test"], config, device))

    if args.mode in {"all", "federated"}:
        fed_model, fed_rounds = run_federated(
            model,
            prepared["clients"],
            prepared["validation"],
            prepared["test"],
            config,
            device,
        )
        fed_metrics = evaluate_model(
            fed_model,
            prepared["test"],
            config["federated"]["batch_size"],
            device,
        )
        fed_row = {
            "experiment": "federated_best",
            "client_id": "global",
            "num_train_windows": sum(len(client["train"]) for client in prepared["clients"].values()),
            "train_loss": float(fed_rounds["train_loss"].iloc[-1]) if not fed_rounds.empty else float("nan"),
        }
        fed_row.update(fed_metrics)
        summary_frames.append(pd.DataFrame([fed_row]))

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary.to_csv(out_dir / "summary_results.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
