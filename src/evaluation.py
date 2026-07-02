"""Evaluation metrics for binary intrusion detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def false_positive_rate(tn: int, fp: int) -> float:
    """Calculate FPR as FP / (FP + TN)."""
    denominator = fp + tn
    return float(fp / denominator) if denominator else 0.0


def _safe_auc(metric_func, y_true, y_score) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(metric_func(y_true, y_score))
    except ValueError:
        return None


def calculate_metrics(y_true, y_pred, y_score=None) -> dict[str, float | int | None]:
    """Calculate standard binary classification metrics."""
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": false_positive_rate(int(tn), int(fp)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_score),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_attack_recall(y_true, y_pred, attack_category) -> pd.DataFrame:
    """Calculate recall for each attack category present in y_true."""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "attack_category": pd.Series(attack_category).astype(str).str.lower().to_numpy(),
        }
    )
    rows: list[dict[str, object]] = []
    for attack, group in frame[frame["y_true"] == 1].groupby("attack_category"):
        positives = len(group)
        detected = int((group["y_pred"] == 1).sum())
        rows.append(
            {
                "attack_category": attack,
                "attack_samples": positives,
                "detected": detected,
                "recall": float(detected / positives) if positives else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("attack_category") if rows else pd.DataFrame()


def threshold_metrics(y_true, scores, thresholds) -> pd.DataFrame:
    """Evaluate metrics and expected false alerts for score thresholds."""
    rows: list[dict[str, object]] = []
    y_true_array = np.asarray(y_true).astype(int)
    score_array = np.asarray(scores)
    for threshold in thresholds:
        y_pred = (score_array >= threshold).astype(int)
        metrics = calculate_metrics(y_true_array, y_pred, score_array)
        for attack_rate in [0.01, 0.001, 0.0001]:
            expected_false_alerts = metrics["false_positive_rate"] * (1 - attack_rate) * 100000
            rows.append(
                {
                    "threshold": threshold,
                    "attack_rate": attack_rate,
                    "expected_false_alerts_per_100k_flows": expected_false_alerts,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
