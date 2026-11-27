
import yaml
import torch

from novel_ids.representation.feature_identity import (
    FeatureIdentityEncoder,
)


DESCRIPTOR_PATH = (
    "src/novel_ids/onboarding/feature_descriptors_with_stats.yaml"
)


def test_level_one_equivalent_features_have_identical_identities():
    with open(DESCRIPTOR_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    descriptors = data["features"]

    level_one = [
        item for item in descriptors
        if item.get("level") == 1
        and item.get("semantic_id")
    ]

    assert len(level_one) == 10

    encoder = FeatureIdentityEncoder()

    encoder.fit(level_one)

    identities = encoder.encode(level_one)

    semantic_groups = {}

    for descriptor, identity in zip(level_one, identities):
        semantic_id = descriptor["semantic_id"]

        semantic_groups.setdefault(
            semantic_id, []
        ).append(identity)

    assert len(semantic_groups) == 5

    for semantic_id, vectors in semantic_groups.items():
        assert len(vectors) == 2, (
            f"Expected two dataset mappings for {semantic_id}"
        )

        assert torch.allclose(
            vectors[0],
            vectors[1],
            atol=1e-7,
            rtol=0,
        ), f"Identity mismatch for {semantic_id}"
