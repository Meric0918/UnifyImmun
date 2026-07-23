"""Binary classification metrics shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass
class BinaryMetrics:
    loss: float
    auroc: float
    aupr: float
    accuracy: float
    mcc: float
    f1: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_binary_metrics(
    labels,
    probabilities,
    *,
    loss: float,
    threshold: float = 0.5,
) -> BinaryMetrics:
    try:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for training metrics. "
            "Install requirements-plm.txt."
        ) from exc

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and probabilities must be aligned 1-D arrays")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC/AUPR require both classes in the evaluation set")
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    return BinaryMetrics(
        loss=float(loss),
        auroc=float(roc_auc_score(y_true, y_score)),
        aupr=float(auc(recall_curve, precision_curve)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        sensitivity=float(tp / (tp + fn)) if tp + fn else 0.0,
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        samples=int(len(y_true)),
    )


def joint_score(
    phla_metrics: BinaryMetrics | Mapping[str, float],
    ptcr_metrics: BinaryMetrics | Mapping[str, float],
) -> float:
    def value(metrics, name: str) -> float:
        if isinstance(metrics, BinaryMetrics):
            return float(getattr(metrics, name))
        return float(metrics[name])

    return (
        value(phla_metrics, "auroc")
        + value(phla_metrics, "aupr")
        + value(ptcr_metrics, "auroc")
        + value(ptcr_metrics, "aupr")
    ) / 4.0


def prefixed_metrics(prefix: str, metrics: BinaryMetrics) -> dict[str, float | int]:
    return {f"{prefix}/{key}": value for key, value in metrics.to_dict().items()}
