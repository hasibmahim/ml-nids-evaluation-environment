"""Model factory functions."""

from __future__ import annotations

from typing import Iterable

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "isolation_forest"]


def get_model(name: str, random_state: int = 42):
    """Create one model by name."""
    key = name.lower()
    if key == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for the xgboost model. Install requirements.txt.") from exc
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
    if key == "isolation_forest":
        return IsolationForest(n_estimators=120, contamination="auto", random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model {name!r}. Supported models: {MODEL_NAMES}")


def get_models(model_names: Iterable[str], random_state: int = 42) -> dict[str, object]:
    """Create multiple models."""
    return {name: get_model(name, random_state=random_state) for name in model_names}


def predict_binary(model, X, model_name: str):
    """Return binary predictions where 1 means attack."""
    if model_name == "isolation_forest":
        raw = model.predict(X)
        return (raw == -1).astype(int)
    return model.predict(X).astype(int)


def predict_scores(model, X, model_name: str):
    """Return anomaly/attack scores where larger values mean more suspicious."""
    if model_name == "isolation_forest":
        return -model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
