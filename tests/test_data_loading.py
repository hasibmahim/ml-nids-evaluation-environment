import pandas as pd

from src.data_loading import load_cicids2017


def test_cicids2017_max_rows_samples_from_multiple_files(tmp_path):
    for file_name, label in [
        ("day1.csv", "BENIGN"),
        ("day2.csv", "DDoS"),
        ("day3.csv", "PortScan"),
    ]:
        pd.DataFrame(
            {
                "Flow Duration": [10, 20, 30, 40],
                "Total Fwd Packets": [1, 2, 3, 4],
                "Label": [label] * 4,
            }
        ).to_csv(tmp_path / file_name, index=False)

    loaded = load_cicids2017(tmp_path, max_rows=6, random_state=123)

    assert len(loaded) == 6
    assert {"benign", "ddos", "portscan"}.issubset(set(loaded["attack_category"]))
