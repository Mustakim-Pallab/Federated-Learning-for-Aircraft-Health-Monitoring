from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        diff = pred - true
        if diff < 0:
            score += math.exp(-diff / 13.0) - 1.0
        else:
            score += math.exp(diff / 10.0) - 1.0
    return float(score)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    labels = np.unique(y_true)
    auroc = float("nan") if len(labels) < 2 else float(roc_auc_score(y_true, y_score))
    auprc = float("nan") if len(labels) < 2 else float(average_precision_score(y_true, y_score))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def all_metrics(
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    fault_true: np.ndarray,
    fault_score: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(rul_true, rul_pred)
    metrics.update(classification_metrics(fault_true.astype(int), fault_score))
    return metrics
