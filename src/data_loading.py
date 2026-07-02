"""Dataset loading and normalization for CIC-IDS2017 and UNSW-NB15."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sample_data import generate_sample_dataset
from src.utils import find_files, normalize_column_name


NON_FEATURE_HINTS = {
    "id",
    "flow_id",
    "timestamp",
    "time",
    "date",
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "source_port",
    "destination_port",
    "proto",
    "protocol",
    "state",
    "service",
}


def _read_csv_folder(folder: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    files = find_files(folder, "*.csv")
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. Place raw dataset CSV files there or use --sample."
        )

    frames: list[pd.DataFrame] = []
    remaining = max_rows
    for file_path in files:
        nrows = remaining if remaining is not None else None
        frame = pd.read_csv(file_path, nrows=nrows, low_memory=False)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break

    if not frames:
        raise ValueError(f"No rows loaded from {folder}")
    return pd.concat(frames, ignore_index=True)


def _normalize_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    return normalized


def _label_to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    text = series.astype(str).str.strip().str.lower()
    benign_values = {"benign", "normal", "0", "none"}
    return (~text.isin(benign_values)).astype(int)


def _prepare_common_columns(
    df: pd.DataFrame,
    label_candidates: list[str],
    attack_candidates: list[str],
) -> pd.DataFrame:
    normalized = _normalize_frame_columns(df)

    label_column = next((name for name in label_candidates if name in normalized.columns), None)
    if label_column is None:
        raise ValueError(f"Could not find a label column. Tried: {label_candidates}")

    attack_column = next((name for name in attack_candidates if name in normalized.columns), None)

    prepared = normalized.copy()
    prepared["target"] = _label_to_binary(prepared[label_column])
    if attack_column:
        prepared["attack_category"] = prepared[attack_column].astype(str).str.strip().str.lower()
    else:
        prepared["attack_category"] = np.where(prepared["target"] == 1, "attack", "benign")

    prepared.loc[prepared["target"] == 0, "attack_category"] = "benign"
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    return prepared


def _drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["target", "attack_category"]
    feature_columns: list[str] = []
    for column in df.columns:
        if column in keep:
            continue
        if column in NON_FEATURE_HINTS:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        feature_columns.append(column)
    return df[feature_columns + keep].copy()


def load_cicids2017(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load CIC-IDS2017 CSV files from a folder."""
    raw = _read_csv_folder(path, max_rows=max_rows)
    prepared = _prepare_common_columns(
        raw,
        label_candidates=["label", "labels", "class"],
        attack_candidates=["attack_category", "attack_cat", "label"],
    )
    return _drop_non_feature_columns(prepared)


def load_unsw_nb15(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load UNSW-NB15 CSV files from a folder."""
    raw = _read_csv_folder(path, max_rows=max_rows)
    prepared = _prepare_common_columns(
        raw,
        label_candidates=["label", "target", "binary_label", "class"],
        attack_candidates=["attack_cat", "attack_category", "category"],
    )
    return _drop_non_feature_columns(prepared)


def load_dataset(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
    sample: bool = False,
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a named dataset, optionally using synthetic sample data."""
    name = dataset_name.lower()
    if sample:
        n_rows = max_rows if max_rows is not None else 600
        return generate_sample_dataset(name, n_rows=n_rows, random_state=random_state)

    base = Path(raw_root)
    if name == "cicids2017":
        return load_cicids2017(base / "cicids2017", max_rows=max_rows)
    if name == "unsw_nb15":
        return load_unsw_nb15(base / "unsw_nb15", max_rows=max_rows)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
