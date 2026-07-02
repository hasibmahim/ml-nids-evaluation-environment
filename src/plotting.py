"""Simple matplotlib plots for thesis-safe result summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def _save_current(path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Plot F1-score by model."""
    if metrics_df.empty or "f1" not in metrics_df:
        return None
    plot_df = metrics_df.groupby("model", as_index=False)["f1"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(plot_df["model"], plot_df["f1"], color="#4c78a8")
    plt.ylabel("F1-score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    return _save_current(output_path)


def plot_fpr_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Plot false positive rate by model."""
    if metrics_df.empty or "false_positive_rate" not in metrics_df:
        return None
    plot_df = metrics_df.groupby("model", as_index=False)["false_positive_rate"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(plot_df["model"], plot_df["false_positive_rate"], color="#f58518")
    plt.ylabel("False positive rate")
    plt.xlabel("Model")
    plt.ylim(0, max(0.05, min(1.0, float(plot_df["false_positive_rate"].max()) * 1.2)))
    plt.xticks(rotation=25, ha="right")
    return _save_current(output_path)


def plot_threshold_curve(threshold_df: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Plot F1 and FPR across thresholds."""
    if threshold_df.empty:
        return None
    base = threshold_df.drop_duplicates(["model", "threshold"]) if "model" in threshold_df else threshold_df
    plt.figure(figsize=(7, 4))
    for model, group in base.groupby("model"):
        plt.plot(group["threshold"], group["f1"], marker="o", label=f"{model} F1")
        plt.plot(group["threshold"], group["false_positive_rate"], marker="x", linestyle="--", label=f"{model} FPR")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    return _save_current(output_path)


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, output_path: str | Path) -> Path:
    """Plot a two-by-two confusion matrix using matplotlib only."""
    matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Pred benign", "Pred attack"], rotation=20, ha="right")
    plt.yticks([0, 1], ["True benign", "True attack"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    return _save_current(output_path)
