# Federated Learning for Aircraft Engine Health Monitoring

Submitted by: **A. S. M Mustakim Rahman Siddique**

Repository: <https://github.com/Mustakim-Pallab/Federated-Learning-for-Aircraft-Health-Monitoring>

This repository implements a federated learning pipeline for aircraft engine prognostics and health management using NASA C-MAPSS turbofan engine data. The system simulates multiple airline operators as federated clients and trains a dual-head neural network for:

- Remaining Useful Life (RUL) regression.
- Early fault detection.

The main technical report is available in:

```text
Report of Project for PhD Applicants.pdf
```

It contains the methodology, implementation details, equations, result tables, limitations, citations, and future research directions.

## Project Scope

- **Task 1:** baseline isolated, centralized, and federated training.
- **Task 2:** all seven research-question experiments from the project brief.
- **Task 3:** future research directions included in the main PDF report.

Implemented research questions:

- **RQ1:** heterogeneous sensor availability and local sensor adapters.
- **RQ2:** class imbalance across clients.
- **RQ3:** root-cause interpretability with input-gradient attribution.
- **RQ4:** concept drift simulation and detection.
- **RQ5:** validation mismatch under non-IID clients.
- **RQ6:** membership inference from client updates.
- **RQ7:** model poisoning and robust aggregation defenses.

## Repository Structure

```text
configs/              YAML configuration files for baseline and RQ experiments
data/                 C-MAPSS loading, preprocessing, windowing, and client splits
models/               Dual-head CNN-GRU and heterogeneous sensor adapter models
federated/            Client training, server aggregation, and FL runner
baselines/            Isolated and centralized baselines
research_questions/   RQ1-RQ7 implementations
experiments/          Experiment entry points and generated logs
evaluation/           Metrics and reporting helpers
src/                  CLI entrypoint used by main.py
Report of Project for PhD Applicants.pdf
                      Main technical report for submission
run_all.sh            Runs Task 1 and RQ1-RQ7 sequentially
```

## Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages are listed in `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `PyYAML`

## Dataset

The experiments use NASA C-MAPSS `FD001`.

Download and extract the C-MAPSS data so these files are available:

```text
data/raw/CMAPSSData/train_FD001.txt
data/raw/CMAPSSData/test_FD001.txt
data/raw/CMAPSSData/RUL_FD001.txt
```

The config files also include the dataset URL. To allow the code to download the dataset automatically, set:

```yaml
dataset:
  download: true
```

Default dataset settings:

- 5 simulated airline clients.
- Engine-level train/validation/client splits to avoid trajectory leakage.
- Sliding windows of length 30 and stride 1.
- Capped RUL target: `min(raw_rul, 125)`.
- Fault target: `1` when raw RUL is less than or equal to 30 cycles.
- Selected sensors: `s2`, `s3`, `s4`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`.

## Model

The baseline model is a dual-head CNN-GRU:

- A shared CNN-GRU encoder processes each 30-cycle sensor window.
- The RUL head predicts a scalar remaining-life value.
- The fault head predicts a binary fault-risk logit.

Training uses a multitask loss:

```text
total_loss = RUL_MSE + fault_loss_weight * BCEWithLogits(fault_logit, fault_label)
```

Federated training uses sample-weighted FedAvg.

## How to Run the Experiments

Run Task 1 and all implemented Task 2 experiments:

```bash
./run_all.sh
```

This executes:

```text
Task 1 baseline
RQ1 variable sensors
RQ2 class imbalance
RQ3 interpretability
RQ4 concept drift
RQ5 validation mismatch
RQ6 membership inference
RQ7 model poisoning
```

All generated CSV outputs are written to:

```text
experiments/logs/
```

Run only Task 1:

```bash
python3 main.py --config configs/task1_baseline.yaml --mode all
```

Run individual research-question experiments:

```bash
python3 experiments/run_rq1.py --config configs/task2_rq1_variable_sensors.yaml
python3 experiments/run_rq2.py --config configs/task2_rq2_class_imbalance.yaml
python3 experiments/run_rq3.py --config configs/task2_rq3_interpretability.yaml
python3 experiments/run_rq4.py --config configs/task2_rq4_concept_drift.yaml
python3 experiments/run_rq5.py --config configs/task2_rq5_validation_mismatch.yaml
python3 experiments/run_rq6.py --config configs/task2_rq6_membership_inference.yaml
python3 experiments/run_rq7.py --config configs/task2_rq7_model_poisoning.yaml
```

## Key Results

Task 1 baseline summary:

| Method | RMSE | MAE | AUROC | AUPRC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolated client mean | 39.524 | 35.906 | 0.9785 | 0.7107 | 0.6377 | 0.6380 | 0.5866 |
| Centralized model | 16.167 | 11.693 | 0.9964 | 0.8999 | 0.7881 | 0.8404 | 0.8134 |
| Federated validation-best model | **14.024** | **10.191** | **0.9972** | **0.9207** | **0.8821** | 0.7440 | 0.8072 |

Selected Task 2 findings:

- **RQ1:** the lowest-common-sensor baseline outperformed the current variable-sensor adapter run, showing that adapter personalization needs stronger regularization or pretraining.
- **RQ2:** weighted BCE increased fault recall up to 0.9669, but reduced precision.
- **RQ3:** input-gradient attribution highlighted bypass/flow-path and thermal-core sensors as important for fault predictions.
- **RQ4:** the configured drift trigger did not detect the injected drift, which is a useful negative result.
- **RQ5:** distribution-aware validation improved precision but did not improve overall RMSE, AUPRC, recall, or F1 in the current run.
- **RQ6:** membership signal was measurable from client updates; the implemented clipping/noise defenses are DP-inspired only and do not provide a formal privacy budget.
- **RQ7:** model poisoning sharply reduced fault recall; the detection-filter defense recovered most of the clean-model near-failure behavior.

Full result tables and interpretation are in `Report of Project for PhD Applicants.pdf`.

## Main Output Files

Task 1:

- `experiments/logs/isolated_results.csv`
- `experiments/logs/centralized_results.csv`
- `experiments/logs/federated_rounds.csv`
- `experiments/logs/summary_results.csv`

Task 2 summaries:

- `experiments/logs/rq1_summary_results.csv`
- `experiments/logs/rq2_summary_results.csv`
- `experiments/logs/rq3_summary_results.csv`
- `experiments/logs/rq4_summary_results.csv`
- `experiments/logs/rq5_summary_results.csv`
- `experiments/logs/rq6_privacy_utility_summary.csv`
- `experiments/logs/rq7_summary_results.csv`

Additional per-round diagnostics, attribution files, validation matrices, update norms, and detection flags are also stored under `experiments/logs/`.

## Notes

- Config files use `device: auto`, which selects CUDA when available and CPU otherwise.
- Most experiments are configured for 30 FL rounds by default.
- SMOTE and KMeans-SMOTE in RQ2 are exploratory baselines. Since C-MAPSS samples are sequential degradation windows, synthetic interpolation may be less physically realistic than real run-to-failure trajectories.
- RQ6 clipping and noise are DP-inspired defenses, but the implementation does not claim a formal differential-privacy budget.
- Empty fields in RQ6 summary CSVs are attack-specific `not applicable` values. For example, loss-threshold rows do not have update-effect loss-drop columns, and update-effect rows do not have loss-threshold mean-loss columns.
- This is a research prototype and not aviation-certified decision support.
