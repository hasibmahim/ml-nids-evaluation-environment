import pandas as pd

from src.evaluation import calculate_metrics, false_positive_rate, per_attack_recall, threshold_metrics


def test_calculate_metrics_confusion_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.8, 0.4, 0.9]

    metrics = calculate_metrics(y_true, y_pred, y_score)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


def test_false_positive_rate_zero_denominator():
    assert false_positive_rate(0, 0) == 0.0


def test_per_attack_recall():
    result = per_attack_recall(
        y_true=[0, 1, 1],
        y_pred=[0, 1, 0],
        attack_category=pd.Series(["benign", "dos", "dos"]),
    )

    assert result.iloc[0]["attack_category"] == "dos"
    assert result.iloc[0]["recall"] == 0.5


def test_threshold_metrics_returns_base_rate_rows():
    result = threshold_metrics([0, 1], [0.2, 0.8], [0.5])

    assert len(result) == 3
    assert "expected_false_alerts_per_100k_flows" in result.columns
