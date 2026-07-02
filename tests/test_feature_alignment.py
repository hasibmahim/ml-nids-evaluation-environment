import pandas as pd
import pytest

from src.feature_alignment import align_features


def test_feature_alignment_manual_mapping():
    cic = pd.DataFrame(
        {
            "flow_duration": [1, 2],
            "total_fwd_packets": [3, 4],
            "total_backward_packets": [5, 6],
            "total_length_of_fwd_packets": [7, 8],
        }
    )
    unsw = pd.DataFrame(
        {
            "dur": [1, 2],
            "spkts": [3, 4],
            "dpkts": [5, 6],
            "sbytes": [7, 8],
        }
    )

    train, test, features, warning = align_features(cic, unsw, "cicids2017", "unsw_nb15")

    assert train.shape == test.shape
    assert "duration" in features
    assert len(features) >= 3
    assert warning is not None


def test_feature_alignment_fails_with_too_few_features():
    cic = pd.DataFrame({"flow_duration": [1, 2]})
    unsw = pd.DataFrame({"dur": [1, 2]})

    with pytest.raises(ValueError):
        align_features(cic, unsw, "cicids2017", "unsw_nb15")
