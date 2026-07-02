"""YAML config loading helpers."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "experiment_name",
    "experiment_type",
    "train_dataset",
    "test_dataset",
    "target_mode",
    "models",
    "max_rows",
    "random_state",
    "output_dir",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"Config {config_path} is missing required keys: {missing}")

    config["_config_path"] = str(config_path)
    return config


def load_configs(configs_dir: str | Path, pattern: str = "*.yaml") -> list[dict[str, Any]]:
    """Load all config files matching a shell-style pattern."""
    base = Path(configs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Config directory not found: {base}")

    paths = sorted(path for path in base.glob("*.yaml") if fnmatch(path.name, pattern))
    if not paths:
        raise FileNotFoundError(f"No config files matched pattern {pattern!r} in {base}")

    return [load_config(path) for path in paths]
