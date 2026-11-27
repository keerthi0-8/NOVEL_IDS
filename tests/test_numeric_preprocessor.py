
import numpy as np
import pandas as pd
import pytest

from src.novel_ids.preprocessing.numeric_preprocessor import (
    NumericPreprocessor,
)


def test_training_only_fit_and_transform():
    X_train = pd.DataFrame({
        "bytes": [10.0, 20.0, 30.0, np.nan],
        "duration": [1.0, 2.0, 3.0, 4.0],
    })

    X_validation = pd.DataFrame({
        "bytes": [np.inf, 1000.0],
        "duration": [5.0, 6.0],
    })

    processor = NumericPreprocessor(log_features=["bytes"])

    X_train_scaled, M_train = processor.fit_transform(X_train)
    X_val_scaled, M_val = processor.transform(X_validation)

    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_validation.shape

    assert M_train.loc[3, "bytes"] == 1.0
    assert M_val.loc[0, "bytes"] == 1.0
    assert M_val.loc[1, "bytes"] == 0.0

    assert np.isfinite(X_train_scaled.to_numpy()).all()
    assert np.isfinite(X_val_scaled.to_numpy()).all()

    # Validation values must not change training-fitted medians.
    assert processor.medians["bytes"] == 20.0
    # The training row missing bytes receives the training median before scaling;
    # the median maps to the robust-scaled center (zero).
    assert X_train_scaled.loc[3, "bytes"] == pytest.approx(0.0)
    assert X_val_scaled.loc[0, "bytes"] == pytest.approx(0.0)
    assert processor.medians["bytes"] == 20.0


def test_reject_negative_log_feature():
    X_train = pd.DataFrame({
        "bytes": [10.0, -2.0, 30.0],
    })

    processor = NumericPreprocessor(log_features=["bytes"])

    with pytest.raises(ValueError):
        processor.fit(X_train)


def test_reject_categorical_input():
    X_train = pd.DataFrame({
        "bytes": [10.0, 20.0],
        "protocol": ["tcp", "udp"],
    })

    processor = NumericPreprocessor()

    with pytest.raises(ValueError):
        processor.fit(X_train)


def test_reject_different_feature_order():
    X_train = pd.DataFrame({
        "bytes": [10.0, 20.0],
        "duration": [1.0, 2.0],
    })

    processor = NumericPreprocessor()
    processor.fit(X_train)

    X_wrong_order = X_train[["duration", "bytes"]]

    with pytest.raises(ValueError):
        processor.transform(X_wrong_order)


def test_reject_entirely_missing_training_feature():
    X_train = pd.DataFrame({
        "bytes": [np.nan, np.nan],
    })

    processor = NumericPreprocessor()

    with pytest.raises(ValueError):
        processor.fit(X_train)
