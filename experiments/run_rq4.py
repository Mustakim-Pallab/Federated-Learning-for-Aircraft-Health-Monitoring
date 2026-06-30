from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_questions.rq4_concept_drift import run_rq4_concept_drift_experiment
from src.cli import load_config, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ4 concept-drift FL experiment.")
    parser.add_argument("--config", default="configs/task2_rq4_concept_drift.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config)
    summary = run_rq4_concept_drift_experiment(config, device)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
