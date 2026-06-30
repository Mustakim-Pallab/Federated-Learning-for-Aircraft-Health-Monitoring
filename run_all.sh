#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_step() {
  local label="$1"
  shift

  printf '\n[%s] Running: %s\n' "$label" "$*"
  "$@"
}

run_step "1/8 Task 1" python3 main.py --config configs/task1_baseline.yaml --mode all
run_step "2/8 RQ1" python3 experiments/run_rq1.py --config configs/task2_rq1_variable_sensors.yaml
run_step "3/8 RQ2" python3 experiments/run_rq2.py --config configs/task2_rq2_class_imbalance.yaml
run_step "4/8 RQ3" python3 experiments/run_rq3.py --config configs/task2_rq3_interpretability.yaml
run_step "5/8 RQ4" python3 experiments/run_rq4.py --config configs/task2_rq4_concept_drift.yaml
run_step "6/8 RQ5" python3 experiments/run_rq5.py --config configs/task2_rq5_validation_mismatch.yaml
run_step "7/8 RQ6" python3 experiments/run_rq6.py --config configs/task2_rq6_membership_inference.yaml
run_step "8/8 RQ7" python3 experiments/run_rq7.py --config configs/task2_rq7_model_poisoning.yaml

printf '\nAll implemented Task 1 and Task 2 experiments completed.\n'