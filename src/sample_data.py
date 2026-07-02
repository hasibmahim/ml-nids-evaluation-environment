"""Synthetic sample data for smoke tests only.

These generators imitate a small subset of flow-level intrusion-detection
features. They are not derived from CIC-IDS2017 or UNSW-NB15 and must not be
reported as thesis results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ATTACKS = ["dos", "probe", "bruteforce", "web_attack"]


def _make_binary_labels(n_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    target = rng.choice([0, 1], size=n_rows, p=[0.72, 0.28])
    attack_category = np.array(["benign"] * n_rows, dtype=object)
    attack_category[target == 1] = rng.choice(ATTACKS, size=int(target.sum()))
    return target, attack_category


def generate_cicids2017_sample(n_rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generate CIC-IDS2017-like synthetic flow data."""
    rng = np.random.default_rng(random_state)
    target, attack_category = _make_binary_labels(n_rows, rng)
    signal = target.astype(float)

    df = pd.DataFrame(
        {
            "flow_duration": rng.gamma(2.0 + signal, 1200.0, n_rows),
            "total_fwd_packets": rng.poisson(8 + 6 * signal, n_rows),
            "total_backward_packets": rng.poisson(6 + 4 * signal, n_rows),
            "total_length_of_fwd_packets": rng.gamma(2.0 + signal, 450.0, n_rows),
            "total_length_of_bwd_packets": rng.gamma(2.0 + signal, 360.0, n_rows),
            "fwd_packet_length_mean": rng.normal(72 + 35 * signal, 12, n_rows),
            "bwd_packet_length_mean": rng.normal(68 + 25 * signal, 10, n_rows),
            "flow_packets_s": rng.gamma(1.5 + signal, 8.0, n_rows),
            "flow_bytes_s": rng.gamma(2.0 + signal, 700.0, n_rows),
            "packet_length_std": rng.normal(22 + 15 * signal, 5, n_rows),
            "target": target,
            "attack_category": attack_category,
        }
    )
    return df


def generate_unsw_nb15_sample(n_rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generate UNSW-NB15-like synthetic flow data."""
    rng = np.random.default_rng(random_state + 17)
    target, attack_category = _make_binary_labels(n_rows, rng)
    signal = target.astype(float)

    df = pd.DataFrame(
        {
            "dur": rng.gamma(2.0 + signal, 1.8, n_rows),
            "spkts": rng.poisson(7 + 6 * signal, n_rows),
            "dpkts": rng.poisson(5 + 5 * signal, n_rows),
            "sbytes": rng.gamma(2.2 + signal, 420.0, n_rows),
            "dbytes": rng.gamma(2.0 + signal, 390.0, n_rows),
            "smean": rng.normal(70 + 35 * signal, 13, n_rows),
            "dmean": rng.normal(66 + 28 * signal, 11, n_rows),
            "rate": rng.gamma(1.6 + signal, 7.5, n_rows),
            "sload": rng.gamma(2.0 + signal, 650.0, n_rows),
            "tcprtt": rng.normal(0.12 + 0.06 * signal, 0.03, n_rows),
            "target": target,
            "attack_category": attack_category,
        }
    )
    return df


def generate_sample_dataset(dataset_name: str, n_rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Return the requested synthetic dataset."""
    key = dataset_name.lower()
    if key == "cicids2017":
        return generate_cicids2017_sample(n_rows=n_rows, random_state=random_state)
    if key == "unsw_nb15":
        return generate_unsw_nb15_sample(n_rows=n_rows, random_state=random_state)
    raise ValueError(f"Unsupported sample dataset: {dataset_name}")
