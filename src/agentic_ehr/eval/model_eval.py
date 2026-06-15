"""Predictive-performance metrics appropriate for binary EHR risk tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    n: int
    positives: int
    auroc: float
    auprc: float
    brier: float
    calibration_error: float   # expected calibration error (ECE)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> ModelMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = _expected_calibration_error(y_true, y_prob, n_bins)
    return ModelMetrics(
        n=len(y_true),
        positives=int(y_true.sum()),
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        calibration_error=ece,
    )


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)
