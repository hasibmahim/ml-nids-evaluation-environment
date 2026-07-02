"""Feature alignment for cross-dataset experiments."""

from __future__ import annotations

import pandas as pd

from src.utils import normalize_column_name


MANUAL_FEATURE_MAPPING: dict[str, dict[str, str]] = {
    "duration": {"cicids2017": "flow_duration", "unsw_nb15": "dur"},
    "src_packets": {"cicids2017": "total_fwd_packets", "unsw_nb15": "spkts"},
    "dst_packets": {"cicids2017": "total_backward_packets", "unsw_nb15": "dpkts"},
    "src_bytes": {"cicids2017": "total_length_of_fwd_packets", "unsw_nb15": "sbytes"},
    "dst_bytes": {"cicids2017": "total_length_of_bwd_packets", "unsw_nb15": "dbytes"},
    "src_packet_mean": {"cicids2017": "fwd_packet_length_mean", "unsw_nb15": "smean"},
    "dst_packet_mean": {"cicids2017": "bwd_packet_length_mean", "unsw_nb15": "dmean"},
    "packet_rate": {"cicids2017": "flow_packets_s", "unsw_nb15": "rate"},
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [normalize_column_name(column) for column in renamed.columns]
    return renamed


def _manual_aligned_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    dataset = dataset_name.lower()
    normalized = _normalize_columns(df)
    data: dict[str, pd.Series] = {}
    for canonical, mapping in MANUAL_FEATURE_MAPPING.items():
        source = normalize_column_name(mapping.get(dataset, ""))
        if source and source in normalized.columns:
            data[canonical] = normalized[source]
    return pd.DataFrame(data, index=df.index)


def align_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]:
    """Align cross-dataset features using manual mapping plus strict intersection."""
    train_norm = _normalize_columns(X_train)
    test_norm = _normalize_columns(X_test)

    manual_train = _manual_aligned_frame(train_norm, train_dataset)
    manual_test = _manual_aligned_frame(test_norm, test_dataset)
    manual_features = [column for column in manual_train.columns if column in manual_test.columns]

    excluded_train = set()
    excluded_test = set()
    for mapping in MANUAL_FEATURE_MAPPING.values():
        train_source = normalize_column_name(mapping.get(train_dataset.lower(), ""))
        test_source = normalize_column_name(mapping.get(test_dataset.lower(), ""))
        if train_source:
            excluded_train.add(train_source)
        if test_source:
            excluded_test.add(test_source)

    strict_features = sorted(
        (set(train_norm.columns) - excluded_train) & (set(test_norm.columns) - excluded_test)
    )

    aligned_train = pd.concat(
        [manual_train[manual_features], train_norm[strict_features]],
        axis=1,
    )
    aligned_test = pd.concat(
        [manual_test[manual_features], test_norm[strict_features]],
        axis=1,
    )
    selected_features = list(aligned_train.columns)

    if len(selected_features) < 3:
        raise ValueError(
            "Cross-dataset evaluation cannot be defended without feature alignment. "
            f"Only {len(selected_features)} shared/mapped features were found. "
            "Check dataset columns or extend MANUAL_FEATURE_MAPPING."
        )

    warning = None
    if len(selected_features) < 8:
        warning = (
            f"Only {len(selected_features)} aligned features are available. "
            "Interpret cross-dataset results carefully."
        )
    return aligned_train, aligned_test, selected_features, warning
