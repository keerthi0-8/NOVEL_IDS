
import numpy as np
import pandas as pd
import pytest

from src.novel_ids.preprocessing.dataset_preprocessor import (
    DatasetPreprocessor,
)


def sample_training_data():
    return pd.DataFrame({
        "bytes": [10.0, 20.0, 30.0, np.nan],
        "duration": [1.0, 2.0, 3.0, 4.0],
        "protocol": ["tcp", "udp", "tcp", "icmp"],
        "service": ["http", "ftp", "http", "smtp"],
    })


def test_combined_preprocessing():
    train = sample_training_data()

    processor = DatasetPreprocessor(log_features=["bytes"])
    result = processor.fit_transform(train)

    assert list(result["numeric_values"].columns) == [
        "bytes", "duration"
    ]
    assert list(result["categorical_values"].columns) == [
        "protocol", "service"
    ]

    assert result["numeric_values"].shape == (4, 2)
    assert result["categorical_values"].shape == (4, 2)

    assert result["numeric_missing_mask"]["bytes"].iloc[3] == 1
    assert np.isfinite(
        result["numeric_values"].to_numpy()
    ).all()


def test_validation_uses_training_parameters():
    train = sample_training_data()

    validation = pd.DataFrame({
        "bytes": [1000.0],
        "duration": [100.0],
        "protocol": ["new_protocol"],
        "service": ["new_service"],
    })

    processor = DatasetPreprocessor(log_features=["bytes"])
    processor.fit(train)

    original_medians = processor.numeric_processor.medians.copy()
    original_vocabularies = (
        processor.categorical_processor.vocabularies.copy()
    )

    result = processor.transform(validation)

    assert processor.numeric_processor.medians.equals(original_medians)
    assert (
        processor.categorical_processor.vocabularies
        == original_vocabularies
    )

    assert result["categorical_values"].iloc[0, 0] == 0
    assert result["categorical_values"].iloc[0, 1] == 0


def test_transform_rejects_wrong_feature_order():
    train = sample_training_data()
    processor = DatasetPreprocessor()
    processor.fit(train)

    with pytest.raises(ValueError):
        processor.transform(train[train.columns[::-1]])


def test_transform_requires_fit():
    processor = DatasetPreprocessor()

    with pytest.raises(RuntimeError):
        processor.transform(sample_training_data())
