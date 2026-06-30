from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_questions.rq1_variable_sensors import run_rq1_suite
from src.cli import load_config, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1 variable-sensor FL experiment.")
    parser.add_argument("--config", default="configs/task2_rq1_variable_sensors.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config)
    summary = run_rq1_suite(config, device)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
