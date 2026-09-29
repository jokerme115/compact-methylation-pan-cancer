from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray, average: str = "macro") -> float | None:
    try:
        return float(roc_auc_score(y_true, y_proba, average=average, multi_class="ovr"))
    except ValueError:
        return None


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | None]:
    labels = np.arange(y_proba.shape[1])
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "roc_auc_ovr_macro": _safe_auc(y_true, y_proba, average="macro"),
        "roc_auc_ovr_weighted": _safe_auc(y_true, y_proba, average="weighted"),
    }
    if y_proba.shape[1] >= 3:
        out["top3_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=3, labels=labels))
    if y_proba.shape[1] >= 5:
        out["top5_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=5, labels=labels))
    return out


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    ids = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ids,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.astype(int),
        }
    )


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    one_hot = np.zeros_like(y_proba, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def uncertainty_frame(y_true: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> pd.DataFrame:
    eps = 1e-12
    entropy = -np.sum(y_proba * np.log(y_proba + eps), axis=1)
    pred_idx = y_proba.argmax(axis=1)
    return pd.DataFrame(
        {
            "true_label": [labels[i] for i in y_true],
            "pred_label": [labels[i] for i in pred_idx],
            "is_correct": pred_idx == y_true,
            "max_probability": y_proba.max(axis=1),
            "entropy": entropy,
            "normalized_entropy": entropy / np.log(max(y_proba.shape[1], 2)),
        }
    )


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric_names: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[BootstrapResult] = []
    n = len(y_true)
    for metric_name in metric_names:
        values: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            metrics = multiclass_metrics(y_true[idx], y_pred[idx], y_proba[idx])
            value = metrics.get(metric_name)
            if value is not None and np.isfinite(value):
                values.append(float(value))
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        rows.append(
            BootstrapResult(
                metric=metric_name,
                mean=float(arr.mean()),
                ci_low=float(np.quantile(arr, 0.025)),
                ci_high=float(np.quantile(arr, 0.975)),
                n_bootstrap=len(values),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    pred = y_proba.argmax(axis=1)
    max_proba = y_proba.max(axis=1)
    correct = (pred == y_true).astype(int)
    return {
        "ece_15_bins": expected_calibration_error(y_true, y_proba, n_bins=15),
        "brier_multiclass": multiclass_brier(y_true, y_proba),
        "confidence_brier": float(brier_score_loss(correct, max_proba)),
        "mean_confidence": float(max_proba.mean()),
        "coverage_at_0_90": float((max_proba >= 0.90).mean()),
        "accuracy_at_0_90": float(correct[max_proba >= 0.90].mean()) if np.any(max_proba >= 0.90) else float("nan"),
    }
