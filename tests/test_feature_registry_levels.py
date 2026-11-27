
from pathlib import Path

import torch
import yaml

from novel_ids.representation.feature_identity import (
    FeatureIdentityEncoder,
)


ROOT = Path(__file__).resolve().parents[1]

DESCRIPTOR_PATH = (
    ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_descriptors_with_stats.yaml"
)


def load_descriptors():
    with open(DESCRIPTOR_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)["features"]


def test_level_one_has_approved_equivalent_mappings():
    descriptors = load_descriptors()
    level_one = [d for d in descriptors if d["level"] == 1]

    assert len(level_one) == 10

    for item in level_one:
        assert item["semantic_id"]
        assert item["relationship"] == "equivalent"
        assert item["review_status"] == "approved"
        assert item["evidence"]


def test_level_two_reviewed_groups_remain_non_equivalent():
    descriptors = load_descriptors()
    level_two = [d for d in descriptors if d["level"] == 2]

    assert len(level_two) == 65
    assert all(item["review_status"] == "approved" for item in level_two)

    for item in level_two:
        assert item["semantic_group"]
        assert item["relationship"] in {"group_only", "partial"}
        assert item["confidence"] in {"high", "medium", "low"}
        assert item["evidence"]
        assert item["review_status"] == "approved"
        assert item["relationship"] != "equivalent"


def test_level_three_remains_dataset_specific():
    descriptors = load_descriptors()
    level_three = [d for d in descriptors if d["level"] == 3]

    assert len(level_three) == 8

    for item in level_three:
        assert item["semantic_id"] is None
        assert item["semantic_group"] is None
        assert item["relationship"] == "dataset_specific"


def test_no_duplicate_dataset_feature_entries():
    descriptors = load_descriptors()

    keys = [
        (d["dataset"], d["original_feature_name"])
        for d in descriptors
    ]

    assert len(keys) == len(set(keys))


def test_unrelated_feature_identities_are_distinct():
    descriptors = load_descriptors()

    encoder = FeatureIdentityEncoder()
    encoder.fit(descriptors)

    first = next(
        d for d in descriptors
        if d["original_feature_name"] == "src_bytes"
    )
    second = next(
        d for d in descriptors
        if d["original_feature_name"] == "protocol_type"
    )

    identities = encoder.encode([first, second])

    assert identities.shape == (2, 128)
    assert torch.isfinite(identities).all()
    assert not torch.allclose(identities[0], identities[1])
