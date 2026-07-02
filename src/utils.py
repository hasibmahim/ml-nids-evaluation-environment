"""Shared utility functions for the evaluation environment."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV, creating parent directories first."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path


def timestamp() -> str:
    """Return a UTC timestamp safe for metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_column_name(name: object) -> str:
    """Normalize dataset column names to a stable snake_case form."""
    text = str(name).strip().lower()
    text = text.replace("/", "_per_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def find_files(folder: str | Path, pattern: str = "*.csv") -> list[Path]:
    """Find matching files in a folder in deterministic order."""
    base = Path(folder)
    if not base.exists():
        return []
    return sorted(path for path in base.glob(pattern) if path.is_file())


def set_random_seed(seed: int) -> None:
    """Set common Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def write_json_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    """Write JSON metadata with readable indentation."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return output_path
