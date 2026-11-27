"""A1 Diagnostic: Verify whether Z_d depends on dataset-specific feature values.

Constructs a valid DualEncoder input, runs in eval mode with no_grad,
changes one available dataset-specific feature value while keeping identity
and mask unchanged, and checks whether Z_d changes.
"""
from __future__ import annotations

import numpy as np
import torch

from novel_ids.representation.dual_encoder import DualEncoder, CATEGORICAL_CONFIG
from novel_ids.representation.feature_identity import FeatureIdentityEncoder


def make_descriptors():
    descriptors = []
    for i in range(78):
        if i < 5:
            item = {
                "original_feature_name": f"common_{i}",
                "metadata_feature_name": f"common_{i}",
                "semantic_id": f"common_{i}",
                "level": 1,
                "semantic_group": None,
                "relationship": "equivalent",
                "confidence": "high",
                "review_status": "approved",
                "data_type": "continuous",
                "statistics": {"skewness": 0.1},
            }
        elif i == 6:
            item = {
                "original_feature_name": "flag",
                "description": "connection state flag",
                "level": 2,
                "semantic_id": None,
                "semantic_group": "connection_state",
                "relationship": "group_only",
                "confidence": "medium",
                "review_status": "approved",
                "data_type": "symbolic",
                "evidence": "Metadata describes connection status.",
                "statistics": {"skewness": 0.2},
            }
        else:
            item = {
                "original_feature_name": f"feature_{i}",
                "description": f"source feature {i}",
                "level": 3,
                "semantic_id": None,
                "semantic_group": None,
                "relationship": "dataset_specific",
                "confidence": "unassessed",
                "review_status": "dataset_specific",
                "data_type": "continuous",
                "statistics": {"skewness": 0.0},
            }
        descriptors.append(item)
    return descriptors


def make_batch(dataset="nsl_kdd", batch_size=2, count=78, descriptors=None):
    config = CATEGORICAL_CONFIG[dataset]
    numeric_mask = torch.ones(count, dtype=torch.bool)
    categorical_mask = torch.zeros(count, dtype=torch.bool)
    for position in config["positions"]:
        numeric_mask[position] = False
        categorical_mask[position] = True
    numeric = torch.zeros(batch_size, count)
    categorical = torch.zeros(batch_size, count, dtype=torch.long)
    for position, size in zip(config["positions"], config["vocab_sizes"]):
        categorical[:, position] = torch.randint(1, size + 1, (batch_size,))
    available = torch.ones(batch_size, count)
    missing = torch.zeros(batch_size, count)
    return {
        "numeric_values": numeric,
        "categorical_values": categorical,
        "M_avail": available,
        "M_miss": missing,
        "numeric_token_mask": numeric_mask,
        "categorical_token_mask": categorical_mask,
        "dataset": dataset,
        "feature_descriptors": descriptors,
    }


def main():
    descriptors = make_descriptors()
    identity_encoder = FeatureIdentityEncoder()
    identity_encoder.fit(descriptors)
    vectors = identity_encoder.encode(descriptors).detach().cpu().numpy()

    model = DualEncoder(
        vectors,
        feature_identity_encoder=identity_encoder,
        source_descriptors=descriptors,
        value_masking_enabled=False,
        feature_set_dropout_rate=0.0,
    )
    model.eval()

    batch = make_batch(batch_size=2, descriptors=descriptors)
    # Use position 10 which is a dataset-specific continuous feature (NSL-KDD positions 1,2,5 are categorical)
    pos = 10
    original_value = batch["numeric_values"].clone()

    with torch.no_grad():
        out1 = model(**batch)

    # Change only the value at position 10, keep identity and mask unchanged
    batch["numeric_values"] = batch["numeric_values"].clone()
    batch["numeric_values"][0, pos] = 12345.0
    batch["numeric_values"][1, pos] = -9999.0

    with torch.no_grad():
        out2 = model(**batch)

    zd_diff = torch.abs(out2["Z_d"] - out1["Z_d"])
    zs_diff = torch.abs(out2["Z_s"] - out1["Z_s"])

    print(f"Position tested: {pos} ({descriptors[pos]['original_feature_name']})")
    print(f"Level: {descriptors[pos]['level']}")
    print(f"Z_d max abs difference: {zd_diff.max().item():.10f}")
    print(f"Z_d mean abs difference: {zd_diff.mean().item():.10f}")
    print(f"Z_d is nonzero diff: {(zd_diff > 0).any().item()}")
    print(f"Z_s max abs difference: {zs_diff.max().item():.10f}")
    print(f"Z_s is nonzero diff: {(zs_diff > 0).any().item()}")
    print(f"\nOriginal Z_d[0]: {out1['Z_d'][0][:5]}")
    print(f"Modified Z_d[0]: {out2['Z_d'][0][:5]}")
    print(f"\nOriginal Z_s[0]: {out1['Z_s'][0][:5]}")
    print(f"Modified Z_s[0]: {out2['Z_s'][0][:5]}")

    if (zd_diff > 0).any():
        print("\nCONCLUSION: Z_d DOES depend on dataset-specific feature values.")
    else:
        print("\nCONCLUSION: Z_d does NOT depend on dataset-specific feature values.")


if __name__ == "__main__":
    main()
