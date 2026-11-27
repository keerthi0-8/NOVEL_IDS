
import torch
import pytest
import numpy as np

from novel_ids.representation.feature_identity import (
    FeatureIdentityEncoder,
)


@pytest.fixture
def source_descriptors():
    return [
        {
            "original_feature_name": "src_bytes",
            "description": "Bytes transferred from source to destination",
            "data_type": "continuous",
            "level": 1,
            "confidence": "high",
            "semantic_group": None,
            "relationship": "equivalent",
            "unit": "bytes",
            "statistics": {
                "median": 120,
                "q99": 5000,
                "zero_fraction": 0.1,
                "distinct_count": 1000,
            },
        },
        {
            "original_feature_name": "protocol_type",
            "description": "Network transport protocol",
            "data_type": "symbolic",
            "level": 1,
            "confidence": "high",
            "semantic_group": None,
            "relationship": "equivalent",
            "unit": None,
            "statistics": {
                "distinct_count": 3,
            },
        },
        {
            "original_feature_name": "flag",
            "description": "Connection status flag",
            "data_type": "symbolic",
            "level": 2,
            "confidence": "medium",
            "semantic_group": "connection_state",
            "relationship": "group_only",
            "unit": None,
            "statistics": {
                "distinct_count": 11,
            },
        },
    ]


def test_encoder_fits_source_descriptors(source_descriptors):
    encoder = FeatureIdentityEncoder()
    encoder.fit(source_descriptors)

    assert encoder.fitted is True


def test_identity_has_expected_dimension(source_descriptors):
    encoder = FeatureIdentityEncoder()
    encoder.fit(source_descriptors)

    output = encoder.encode(source_descriptors)

    assert output.shape == (3, 128)


def test_new_unseen_feature_is_supported(source_descriptors):
    encoder = FeatureIdentityEncoder()
    encoder.fit(source_descriptors)

    unseen_feature = {
        "original_feature_name": "new_packet_interval_metric",
        "description": "Time interval between network packets",
        "data_type": "continuous",
        "level": 3,
        "confidence": "unassessed",
        "semantic_group": None,
        "relationship": "dataset_specific",
        "unit": "milliseconds",
        "statistics": {
            "median": 0.25,
            "q99": 12.5,
            "zero_fraction": 0.02,
            "distinct_count": 500,
        },
    }

    output = encoder.encode([unseen_feature])

    assert output.shape == (1, 128)
    assert torch.isfinite(output).all()


def test_unrelated_descriptors_have_distinct_raw_inputs(
    source_descriptors,
):
    encoder = FeatureIdentityEncoder()
    encoder.fit(source_descriptors)

    first = encoder._numeric_metadata(source_descriptors[0])
    second = encoder._numeric_metadata(source_descriptors[1])

    first_text = encoder._text(source_descriptors[0])
    second_text = encoder._text(source_descriptors[1])

    assert first_text != second_text
    assert not (first == second).all()


def test_identity_projection_receives_gradients(source_descriptors):
    encoder = FeatureIdentityEncoder()
    encoder.fit(source_descriptors)

    output = encoder.encode(source_descriptors)
    loss = output.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in encoder.projection.parameters()
    ]

    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)


def test_encoding_requires_fit(source_descriptors):
    encoder = FeatureIdentityEncoder()

    with pytest.raises(RuntimeError):
        encoder.encode(source_descriptors)


def test_empty_descriptors_are_rejected():
    encoder = FeatureIdentityEncoder()

    with pytest.raises(ValueError):
        encoder.fit([])
def test_binary_feature_type_is_supported():
    encoder = FeatureIdentityEncoder()

    binary_feature = {
        "original_feature_name": "land",
        "description": "Whether the connection is from/to the same host",
        "data_type": "binary",
        "level": 3,
        "confidence": "high",
        "semantic_group": None,
        "relationship": "dataset_specific",
        "unit": None,
        "statistics": {
            "median": 0,
            "distinct_count": 2,
            "is_binary": 1,
        },
    }

    encoder.fit([binary_feature])
    metadata = encoder._numeric_metadata(binary_feature)

    assert metadata.shape[0] == 24
    assert np.isfinite(metadata).all()
    assert metadata[15].item() == 1.0