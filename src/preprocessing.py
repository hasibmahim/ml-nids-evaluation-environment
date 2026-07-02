"""Preprocessing utilities that avoid fitting on test data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature columns with infinite values replaced by NaN."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    return numeric.replace([np.inf, -np.inf], np.nan)


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split features, target, and attack categories using stratification when possible."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X = df.drop(columns=[target_col, "attack_category"], errors="ignore")
    y = df[target_col].astype(int)
    attacks = df.get("attack_category", pd.Series(["unknown"] * len(df), index=df.index))
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    return train_test_split(
        X,
        y,
        attacks,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def remove_constant_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Remove columns that are constant in the training set only."""
    train = clean_numeric_features(X_train)
    test = clean_numeric_features(X_test)
    variable_columns = [column for column in train.columns if train[column].nunique(dropna=False) > 1]
    if not variable_columns:
        raise ValueError("No variable numeric feature columns remain after cleaning.")
    return train[variable_columns], test.reindex(columns=variable_columns), variable_columns


def handle_missing_values(strategy: str = "median") -> SimpleImputer:
    """Return an imputer fitted later inside the training-only pipeline."""
    return SimpleImputer(strategy=strategy)


def build_preprocessor(feature_columns: list[str], standardize: bool = True) -> ColumnTransformer:
    """Build a training-only preprocessing transformer for numeric features."""
    steps = [("imputer", handle_missing_values("median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """Clean, fit preprocessing on training data, and transform train/test."""
    X_train_clean, X_test_clean, feature_columns = remove_constant_columns(X_train, X_test)
    preprocessor = build_preprocessor(feature_columns, standardize=standardize)
    X_train_processed = preprocessor.fit_transform(X_train_clean)
    X_test_processed = preprocessor.transform(X_test_clean)
    return X_train_processed, X_test_processed, preprocessor, feature_columns
