from src.sample_data import generate_cicids2017_sample, generate_unsw_nb15_sample


def test_sample_data_has_required_columns():
    cic = generate_cicids2017_sample(n_rows=50, random_state=1)
    unsw = generate_unsw_nb15_sample(n_rows=50, random_state=1)

    for frame in [cic, unsw]:
        assert "target" in frame.columns
        assert "attack_category" in frame.columns
        assert len(frame) == 50
        assert set(frame["target"].unique()).issubset({0, 1})


def test_sample_data_has_attack_and_benign_rows():
    frame = generate_cicids2017_sample(n_rows=200, random_state=7)

    assert (frame["target"] == 0).any()
    assert (frame["target"] == 1).any()
    assert "benign" in set(frame["attack_category"])
