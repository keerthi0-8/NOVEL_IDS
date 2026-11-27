
from pathlib import Path
import copy

import torch
import yaml

from novel_ids.representation.feature_identity import (
    FeatureIdentityEncoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DESCRIPTOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_descriptors_with_stats.yaml"
)


def load_descriptors():
    with open(DESCRIPTOR_PATH, encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data["features"]


def test_all_83_real_features_have_statistics():
    descriptors = load_descriptors()

    assert len(descriptors) == 83

    for descriptor in descriptors:
        assert "statistics" in descriptor
        assert descriptor["statistics"] is not None


def test_real_feature_identity_generation():
    descriptors = load_descriptors()

    encoder = FeatureIdentityEncoder()
    encoder.fit(descriptors)

    identities = encoder.encode(descriptors)

    assert identities.shape == (83, 128)
    assert torch.isfinite(identities).all()


def test_all_three_feature_levels_are_supported():
    descriptors = load_descriptors()

    levels = {item["level"] for item in descriptors}

    assert levels == {1, 2, 3}


def test_approved_semantic_ids_are_available():
    descriptors = load_descriptors()

    level_one = [
        item for item in descriptors
        if item["level"] == 1
    ]

    assert len(level_one) == 10

    for descriptor in level_one:
        assert descriptor["semantic_id"] is not None


def test_dataset_name_does_not_enter_text_identity():
    descriptors = load_descriptors()

    encoder = FeatureIdentityEncoder()
    encoder.fit(descriptors)

    original = copy.deepcopy(descriptors[0])
    changed = copy.deepcopy(descriptors[0])

    changed["dataset"] = "completely_new_dataset"

    assert encoder._text(original) == encoder._text(changed)

    first = encoder.encode([original])
    second = encoder.encode([changed])

    assert torch.allclose(first, second)


def test_new_feature_works_with_real_fitted_encoder():
    descriptors = load_descriptors()

    encoder = FeatureIdentityEncoder()
    encoder.fit(descriptors)

    new_feature = {
        "dataset": "future_dataset",
        "original_feature_name": "new_flow_interval_metric",
        "metadata_feature_name": "new_flow_interval_metric",
        "semantic_id": None,
        "semantic_group": None,
        "relationship": "dataset_specific",
        "description": "Time interval between network packets",
        "data_type": "continuous",
        "unit": "milliseconds",
        "level": 3,
        "confidence": "unassessed",
        "statistics": {
            "q01": 0.01,
            "q25": 0.10,
            "median": 0.25,
            "q75": 0.50,
            "q99": 12.5,
            "mean": 0.35,
            "std": 0.40,
            "skewness": 2.1,
            "zero_fraction": 0.02,
            "distinct_count": 500,
            "missing_fraction": 0.0,
            "is_binary": False,
            "is_integer": False,
        },
    }

    identity = encoder.encode([new_feature])

    assert identity.shape == (1, 128)
    assert torch.isfinite(identity).all()


def test_real_identity_projection_receives_gradients():
    descriptors = load_descriptors()

    encoder = FeatureIdentityEncoder()
    encoder.fit(descriptors)

    identities = encoder.encode(descriptors)
    loss = identities.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in encoder.projection.parameters()
    ]

    assert all(gradient is not None for gradient in gradients)
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )
