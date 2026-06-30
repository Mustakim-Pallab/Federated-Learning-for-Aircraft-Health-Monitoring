from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.preprocessing import prepare_cmapss_clients
from models.dual_head_model import build_model
from research_questions.rq2_class_imbalance import run_rq2_suite
from src.cli import load_config, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ2 class-imbalance FL experiment.")
    parser.add_argument("--config", default="configs/task2_rq2_class_imbalance.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config)
    prepared = prepare_cmapss_clients(config)
    model = build_model(prepared["input_size"], config)

    summary = run_rq2_suite(
        model,
        prepared["clients"],
        prepared["validation"],
        prepared["test"],
        config,
        device,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
