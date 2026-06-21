# Federated Aircraft Engine RUL Baseline

This repository implements Task 1 from the PhD applicant project brief: a federated learning baseline for aircraft engine health monitoring.

## Best Task 1 Plan

Start with a reliable C-MAPSS baseline.

1. **Dataset**
   - First implementation: C-MAPSS `FD001`.
   - Split complete engine trajectories into `3-6` simulated airline clients.
   - Do not randomly split windows across clients, because that leaks one engine's timeline across airlines.
   - Later extension: N-CMAPSS DS02, which better matches the Landau et al. aircraft FL case study.

2. **Labels**
   - RUL regression target: `max_cycle_for_engine - current_cycle`.
   - Capped RUL target: `min(raw_rul, 125)`.
   - Early fault label: `1 if raw_rul <= 30 else 0`.

3. **Input**
   - Use sliding windows over multivariate sensor time series.
   - Default window size: `30`.
   - Default sensors: the 14 commonly retained C-MAPSS sensors:
     `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.
   - Normalize using only training-engine statistics.

4. **Model**
   - Shared CNN-GRU encoder.
   - RUL regression head.
   - Early fault binary classification head.
   - Loss: `MSE(RUL) + BCE(fault)`.

5. **Training Setups**
   - Isolated baseline: each airline trains alone.
   - Centralized baseline: all data pooled, used as an upper bound.
   - Federated baseline: FedAvg with one central server and `3-6` airline clients.

6. **Metrics**
   - RUL: RMSE, MAE, NASA score.
   - Fault detection: AUPRC, AUROC, precision, recall, F1.
   - Report per-client isolated results and global federated/centralized results.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract the NASA C-MAPSS data so files like `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` are under:

```text
data/raw/CMAPSSData/
```

Alternatively set `dataset.download: true` in `configs/config.yaml`.

## Run

```bash
python3 main.py --mode all
```

Outputs are written to:

```text
experiments/logs/
```

For Colab, use `colab_task1_baseline.py`. It contains the same baseline in one self-contained file.
