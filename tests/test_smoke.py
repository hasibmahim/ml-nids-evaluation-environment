from src.experiments import run_experiment


def test_tiny_sample_within_experiment_smoke(tmp_path):
    config = {
        "experiment_name": "tiny_smoke",
        "experiment_type": "within_dataset",
        "train_dataset": "cicids2017",
        "test_dataset": "cicids2017",
        "target_mode": "binary",
        "models": ["logistic_regression"],
        "max_rows": 80,
        "random_state": 42,
        "output_dir": str(tmp_path),
    }

    metrics = run_experiment(config, sample=True)

    assert len(metrics) == 1
    assert metrics.iloc[0]["model"] == "logistic_regression"
    assert (tmp_path / "tables" / "tiny_smoke_sample_metrics.csv").exists()
    assert (tmp_path / "metadata" / "tiny_smoke_sample_metadata.json").exists()
