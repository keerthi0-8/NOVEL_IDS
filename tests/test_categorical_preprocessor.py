
import pandas as pd
import pytest

from src.novel_ids.preprocessing.categorical_preprocessor import (
    CategoricalPreprocessor,
)


def test_training_categories_are_encoded():
    train = pd.DataFrame({
        "protocol": ["tcp", "udp", "tcp", "icmp"],
        "service": ["http", "ftp", "http", "smtp"],
    })

    processor = CategoricalPreprocessor()
    encoded, missing = processor.fit_transform(train)

    assert encoded.shape == train.shape
    assert encoded["protocol"].nunique() == 3
    assert missing.to_numpy().sum() == 0


def test_unseen_category_uses_unknown_id():
    train = pd.DataFrame({
        "protocol": ["tcp", "udp"],
    })

    validation = pd.DataFrame({
        "protocol": ["tcp", "new_protocol"],
    })

    processor = CategoricalPreprocessor()
    processor.fit(train)

    encoded, _ = processor.transform(validation)

    assert encoded.loc[1, "protocol"] == 0


def test_missing_category_is_tracked():
    train = pd.DataFrame({
        "protocol": ["tcp", None, "udp"],
    })

    processor = CategoricalPreprocessor()
    encoded, missing = processor.fit_transform(train)

    assert encoded.loc[1, "protocol"] == 0
    assert missing.loc[1, "protocol"] == 1
    assert missing.loc[0, "protocol"] == 0


def test_validation_does_not_change_vocabulary():
    train = pd.DataFrame({
        "protocol": ["tcp", "udp"],
    })

    validation = pd.DataFrame({
        "protocol": ["new_protocol"],
    })

    processor = CategoricalPreprocessor()
    processor.fit(train)

    original_vocabulary = processor.vocabularies.copy()
    processor.transform(validation)

    assert processor.vocabularies == original_vocabulary


def test_reject_different_feature_order():
    train = pd.DataFrame({
        "protocol": ["tcp"],
        "service": ["http"],
    })

    processor = CategoricalPreprocessor()
    processor.fit(train)

    wrong_order = train[["service", "protocol"]]

    with pytest.raises(ValueError):
        processor.transform(wrong_order)


def test_transform_requires_fit():
    processor = CategoricalPreprocessor()

    with pytest.raises(RuntimeError):
        processor.transform(pd.DataFrame({"protocol": ["tcp"]}))
