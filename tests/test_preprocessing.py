from src.preprocessing import fit_transform_preprocessor, split_train_test
from src.sample_data import generate_cicids2017_sample


def test_preprocessing_does_not_crash():
    df = generate_cicids2017_sample(n_rows=80, random_state=5)
    X_train, X_test, y_train, y_test, _, _ = split_train_test(df, random_state=5)

    X_train_processed, X_test_processed, preprocessor, features = fit_transform_preprocessor(X_train, X_test)

    assert X_train_processed.shape[0] == len(y_train)
    assert X_test_processed.shape[0] == len(y_test)
    assert len(features) > 0
    assert hasattr(preprocessor, "transform")
