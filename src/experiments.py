"""Experiment implementations and result saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loading import load_dataset
from src.evaluation import calculate_metrics, per_attack_recall, threshold_metrics
from src.feature_alignment import align_features
from src.models import get_model, predict_binary, predict_scores
from src.plotting import plot_fpr_comparison, plot_model_comparison, plot_threshold_curve
from src.preprocessing import fit_transform_preprocessor, split_train_test
from src.utils import save_dataframe, timestamp, write_json_metadata


RAW_ROOT = Path("data/raw")


def _output_name(config: dict[str, Any], sample: bool) -> str:
    name = str(config["experiment_name"])
    return f"{name}_sample" if sample else name


def _paths(config: dict[str, Any], sample: bool) -> dict[str, Path]:
    output_dir = Path(config.get("output_dir", "results"))
    name = _output_name(config, sample)
    return {
        "metrics": output_dir / "tables" / f"{name}_metrics.csv",
        "per_attack": output_dir / "tables" / f"{name}_per_attack_recall.csv",
        "thresholds": output_dir / "tables" / f"{name}_thresholds.csv",
        "f1_plot": output_dir / "figures" / f"{name}_f1_comparison.png",
        "fpr_plot": output_dir / "figures" / f"{name}_fpr_comparison.png",
        "threshold_plot": output_dir / "figures" / f"{name}_threshold_curve.png",
        "metadata": output_dir / "metadata" / f"{name}_metadata.json",
    }


def _standardize_for_model(model_name: str) -> bool:
    return model_name in {"logistic_regression", "xgboost", "isolation_forest"}


def _fit_model(model_name: str, X_train, y_train, random_state: int):
    model = get_model(model_name, random_state=random_state)
    if model_name == "isolation_forest":
        benign_mask = pd.Series(y_train).astype(int).to_numpy() == 0
        fit_X = X_train[benign_mask] if benign_mask.any() else X_train
        model.fit(fit_X)
    else:
        if pd.Series(y_train).nunique() < 2:
            raise ValueError(f"{model_name} requires both benign and attack samples in training data.")
        model.fit(X_train, y_train)
    return model


def _error_metric_row(
    model_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    extra: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Create an explicit row for a model failure."""
    return {
        **extra,
        "model": model_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "false_positive_rate": None,
        "roc_auc": None,
        "pr_auc": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
        "error": str(error),
    }


def _evaluate_models(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    attack_test: pd.Series,
    model_names: list[str],
    random_state: int,
    extra_row_values: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    attack_rows: list[pd.DataFrame] = []
    extra = extra_row_values or {}

    for model_name in model_names:
        try:
            X_train, X_test, _, selected_features = fit_transform_preprocessor(
                X_train_raw,
                X_test_raw,
                standardize=_standardize_for_model(model_name),
            )
            model = _fit_model(model_name, X_train, y_train, random_state)
            y_pred = predict_binary(model, X_test, model_name)
            y_score = predict_scores(model, X_test, model_name)
            metrics = calculate_metrics(y_test, y_pred, y_score)
            metric_rows.append(
                {
                    **extra,
                    "model": model_name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_features": int(len(selected_features)),
                    "error": None,
                    **metrics,
                }
            )

            attack_df = per_attack_recall(y_test, y_pred, attack_test)
            if not attack_df.empty:
                attack_df.insert(0, "model", model_name)
                for key, value in reversed(list(extra.items())):
                    attack_df.insert(0, key, value)
                attack_rows.append(attack_df)
        except Exception as exc:
            print(f"WARNING: {model_name} failed: {exc}")
            metric_rows.append(_error_metric_row(model_name, y_train, y_test, extra, exc))

    per_attack_df = pd.concat(attack_rows, ignore_index=True) if attack_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), per_attack_df


def _save_common_outputs(
    config: dict[str, Any],
    sample: bool,
    metrics_df: pd.DataFrame,
    per_attack_df: pd.DataFrame | None,
    metadata: dict[str, Any],
    thresholds_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    paths = _paths(config, sample)
    written: dict[str, Path] = {}
    written["metrics"] = save_dataframe(metrics_df, paths["metrics"])
    plot_model_comparison(metrics_df, paths["f1_plot"])
    plot_fpr_comparison(metrics_df, paths["fpr_plot"])
    written["f1_plot"] = paths["f1_plot"]
    written["fpr_plot"] = paths["fpr_plot"]

    if per_attack_df is not None and not per_attack_df.empty:
        written["per_attack"] = save_dataframe(per_attack_df, paths["per_attack"])

    if thresholds_df is not None and not thresholds_df.empty:
        written["thresholds"] = save_dataframe(thresholds_df, paths["thresholds"])
        plot_threshold_curve(thresholds_df, paths["threshold_plot"])
        written["threshold_plot"] = paths["threshold_plot"]

    metadata = {
        **metadata,
        "generated_at_utc": timestamp(),
        "sample_mode": sample,
        "sample_warning": "Synthetic demo data only; not thesis results." if sample else None,
        "output_files": {key: str(value) for key, value in written.items()},
    }
    written["metadata"] = write_json_metadata(metadata, paths["metadata"])
    return written


def run_within_dataset_experiment(
    config: dict[str, Any],
    sample: bool = False,
    max_rows: int | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Run a train/test split on one dataset."""
    config = {**config}
    if output_dir:
        config["output_dir"] = output_dir
    random_state = random_state if random_state is not None else int(config["random_state"])
    max_rows = max_rows if max_rows is not None else config.get("max_rows")

    df = load_dataset(config["train_dataset"], RAW_ROOT, sample=sample, max_rows=max_rows, random_state=random_state)
    X_train, X_test, y_train, y_test, _, attack_test = split_train_test(df, random_state=random_state)
    metrics_df, per_attack_df = _evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        attack_test,
        list(config["models"]),
        random_state,
        {"experiment_name": _output_name(config, sample), "experiment_type": config["experiment_type"]},
    )

    _save_common_outputs(
        config,
        sample,
        metrics_df,
        per_attack_df,
        metadata={"config": config, "dataset_rows": int(len(df))},
    )
    return metrics_df


def run_cross_dataset_experiment(
    config: dict[str, Any],
    sample: bool = False,
    max_rows: int | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Train on one dataset and test on another after feature alignment."""
    config = {**config}
    if output_dir:
        config["output_dir"] = output_dir
    random_state = random_state if random_state is not None else int(config["random_state"])
    max_rows = max_rows if max_rows is not None else config.get("max_rows")

    train_df = load_dataset(config["train_dataset"], RAW_ROOT, sample=sample, max_rows=max_rows, random_state=random_state)
    test_df = load_dataset(config["test_dataset"], RAW_ROOT, sample=sample, max_rows=max_rows, random_state=random_state + 1)

    X_train_raw = train_df.drop(columns=["target", "attack_category"], errors="ignore")
    y_train = train_df["target"].astype(int)
    X_test_raw = test_df.drop(columns=["target", "attack_category"], errors="ignore")
    y_test = test_df["target"].astype(int)
    attack_test = test_df["attack_category"]

    X_train_aligned, X_test_aligned, aligned_features, warning = align_features(
        X_train_raw,
        X_test_raw,
        config["train_dataset"],
        config["test_dataset"],
    )
    if warning:
        print(f"WARNING: {warning}")

    metrics_df, per_attack_df = _evaluate_models(
        X_train_aligned,
        X_test_aligned,
        y_train,
        y_test,
        attack_test,
        list(config["models"]),
        random_state,
        {"experiment_name": _output_name(config, sample), "experiment_type": config["experiment_type"]},
    )

    _save_common_outputs(
        config,
        sample,
        metrics_df,
        per_attack_df,
        metadata={
            "config": config,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "aligned_features": aligned_features,
            "alignment_warning": warning,
        },
    )
    return metrics_df


def run_leave_one_attack_out_experiment(
    config: dict[str, Any],
    sample: bool = False,
    max_rows: int | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Hold out one attack category at a time."""
    config = {**config}
    if output_dir:
        config["output_dir"] = output_dir
    random_state = random_state if random_state is not None else int(config["random_state"])
    max_rows = max_rows if max_rows is not None else config.get("max_rows")

    df = load_dataset(config["train_dataset"], RAW_ROOT, sample=sample, max_rows=max_rows, random_state=random_state)
    attacks = sorted(category for category in df["attack_category"].dropna().unique() if category != "benign")
    if not attacks:
        raise ValueError("No attack categories found for leave-one-attack-class-out evaluation.")

    metric_frames: list[pd.DataFrame] = []
    per_attack_frames: list[pd.DataFrame] = []
    for held_out in attacks:
        benign = df[df["attack_category"] == "benign"]
        benign_train, benign_test = split_train_test(
            benign,
            random_state=random_state,
            test_size=0.3,
        )[:2]
        train_df = pd.concat(
            [df[(df["attack_category"] != held_out) & (df["attack_category"] != "benign")], benign_train.assign(target=0, attack_category="benign")],
            ignore_index=True,
        )
        test_df = pd.concat(
            [df[df["attack_category"] == held_out], benign_test.assign(target=0, attack_category="benign")],
            ignore_index=True,
        )

        X_train_raw = train_df.drop(columns=["target", "attack_category"], errors="ignore")
        y_train = train_df["target"].astype(int)
        X_test_raw = test_df.drop(columns=["target", "attack_category"], errors="ignore")
        y_test = test_df["target"].astype(int)
        attack_test = test_df["attack_category"]

        metrics_df, per_attack_df = _evaluate_models(
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            attack_test,
            list(config["models"]),
            random_state,
            {
                "experiment_name": _output_name(config, sample),
                "experiment_type": config["experiment_type"],
                "held_out_attack": held_out,
            },
        )
        metric_frames.append(metrics_df)
        if not per_attack_df.empty:
            per_attack_frames.append(per_attack_df)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_per_attack = pd.concat(per_attack_frames, ignore_index=True) if per_attack_frames else pd.DataFrame()
    _save_common_outputs(
        config,
        sample,
        all_metrics,
        all_per_attack,
        metadata={"config": config, "dataset_rows": int(len(df)), "held_out_attacks": attacks},
    )
    return all_metrics


def run_threshold_analysis(
    config: dict[str, Any],
    sample: bool = False,
    max_rows: int | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate score thresholds and false-alert implications."""
    config = {**config}
    if output_dir:
        config["output_dir"] = output_dir
    random_state = random_state if random_state is not None else int(config["random_state"])
    max_rows = max_rows if max_rows is not None else config.get("max_rows")
    thresholds = config.get("thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    df = load_dataset(config["train_dataset"], RAW_ROOT, sample=sample, max_rows=max_rows, random_state=random_state)
    X_train, X_test, y_train, y_test, _, attack_test = split_train_test(df, random_state=random_state)
    metric_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    per_attack_frames: list[pd.DataFrame] = []

    for model_name in config["models"]:
        extra = {"experiment_name": _output_name(config, sample), "experiment_type": config["experiment_type"]}
        try:
            X_train_processed, X_test_processed, _, selected_features = fit_transform_preprocessor(
                X_train,
                X_test,
                standardize=_standardize_for_model(model_name),
            )
            model = _fit_model(model_name, X_train_processed, y_train, random_state)
            scores = predict_scores(model, X_test_processed, model_name)
            if scores is None:
                raise ValueError("No score output available for threshold analysis.")
            y_pred = predict_binary(model, X_test_processed, model_name)
            metrics = calculate_metrics(y_test, y_pred, scores)
            metric_rows.append(
                {
                    **extra,
                    "model": model_name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_features": int(len(selected_features)),
                    "error": None,
                    **metrics,
                }
            )
            model_thresholds = threshold_metrics(y_test, scores, thresholds)
            model_thresholds.insert(0, "model", model_name)
            model_thresholds.insert(0, "experiment_name", _output_name(config, sample))
            threshold_frames.append(model_thresholds)
            attack_df = per_attack_recall(y_test, y_pred, attack_test)
            if not attack_df.empty:
                attack_df.insert(0, "model", model_name)
                attack_df.insert(0, "experiment_name", _output_name(config, sample))
                per_attack_frames.append(attack_df)
        except Exception as exc:
            print(f"WARNING: {model_name} threshold analysis failed: {exc}")
            metric_rows.append(_error_metric_row(model_name, y_train, y_test, extra, exc))

    metrics_df = pd.DataFrame(metric_rows)
    thresholds_df = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    per_attack_df = pd.concat(per_attack_frames, ignore_index=True) if per_attack_frames else pd.DataFrame()
    _save_common_outputs(
        config,
        sample,
        metrics_df,
        per_attack_df,
        metadata={"config": config, "dataset_rows": int(len(df)), "thresholds": thresholds},
        thresholds_df=thresholds_df,
    )
    return metrics_df


def run_experiment(
    config: dict[str, Any],
    sample: bool = False,
    max_rows: int | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Dispatch to an experiment implementation."""
    experiment_type = config["experiment_type"]
    if experiment_type == "within_dataset":
        return run_within_dataset_experiment(config, sample, max_rows, output_dir, random_state)
    if experiment_type == "cross_dataset":
        return run_cross_dataset_experiment(config, sample, max_rows, output_dir, random_state)
    if experiment_type == "leave_one_attack_out":
        return run_leave_one_attack_out_experiment(config, sample, max_rows, output_dir, random_state)
    if experiment_type == "threshold_analysis":
        return run_threshold_analysis(config, sample, max_rows, output_dir, random_state)
    raise ValueError(f"Unsupported experiment_type: {experiment_type}")
