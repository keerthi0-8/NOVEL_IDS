
from pathlib import Path

import yaml

from load_source_datasets import (
    load_config,
    load_nsl_kdd,
    load_unsw_nb15,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "semantic_feature_registry.yaml"
)


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def separate_features(dataset_name, X, registry):

    common_mapping = {
        item[dataset_name]: item["semantic_id"]
        for item in registry["common_features"]
    }

    common_features = [
        feature for feature in X.columns
        if feature in common_mapping
    ]

    specific_features = [
        feature for feature in X.columns
        if feature not in common_mapping
    ]

    common_data = X[common_features].copy()

    common_data.columns = [
        common_mapping[feature]
        for feature in common_features
    ]

    specific_data = X[specific_features].copy()

    specific_data.columns = [
        f"{dataset_name}::{feature}"
        for feature in specific_features
    ]

    return common_data, specific_data


def main():

    config = load_config()
    registry = load_registry()

    nsl = load_nsl_kdd(config)
    unsw = load_unsw_nb15(config)

    datasets = {
        "nsl_kdd": nsl,
        "unsw_nb15": unsw,
    }

    total_original = 0
    total_specific = 0

    print("\n" + "=" * 65)
    print("NOVEL-IDS COMMON AND DATASET-SPECIFIC FEATURE SEPARATION")
    print("=" * 65)

    expected_common = len(registry["common_features"])

    for name, (X_train, y_train, X_test, y_test) in datasets.items():

        common_train, specific_train = separate_features(
            name, X_train, registry
        )

        common_test, specific_test = separate_features(
            name, X_test, registry
        )

        # Validate common feature count.
        if common_train.shape[1] != expected_common:
            raise ValueError(
                f"{name}: expected {expected_common} common features, "
                f"found {common_train.shape[1]}"
            )

        # Validate train/test feature consistency.
        if list(common_train.columns) != list(common_test.columns):
            raise ValueError(f"{name}: common train/test mismatch")

        if list(specific_train.columns) != list(specific_test.columns):
            raise ValueError(f"{name}: specific train/test mismatch")

        # Check that common and specific identities do not overlap.
        if set(common_train.columns) & set(specific_train.columns):
            raise ValueError(f"{name}: feature identity collision")

        # Verify that every original feature is preserved.
        original_count = X_train.shape[1]

        separated_count = (
            common_train.shape[1] + specific_train.shape[1]
        )

        if original_count != separated_count:
            raise ValueError(
                f"{name}: some original features were lost"
            )

        total_original += original_count
        total_specific += specific_train.shape[1]

        print(f"\n{name.upper()}")
        print("-" * 50)

        print(f"Original features: {original_count}")
        print(f"Common features: {common_train.shape[1]}")
        print(f"Dataset-specific features: {specific_train.shape[1]}")
        print(f"Training rows: {len(X_train):,}")
        print(f"Testing rows: {len(X_test):,}")

        print("Feature preservation: PASS")
        print("Train/test consistency: PASS")

        print("\nCommon semantic identities:")

        for feature in common_train.columns:
            print(f"  - {feature}")

    # Shared common identities are counted only once.
    total_unified = expected_common + total_specific

    expected_original = 41 + 42
    expected_specific = (41 - expected_common) + (
        42 - expected_common
    )

    if total_original != expected_original:
        raise ValueError("Original feature entry count is incorrect")

    if total_specific != expected_specific:
        raise ValueError("Dataset-specific feature count is incorrect")

    if total_unified != 78:
        raise ValueError("Unified feature identity count is incorrect")

    print("\n" + "=" * 65)
    print(f"Original feature entries: {total_original}")
    print(f"Shared common identities: {expected_common}")
    print(f"Dataset-specific identities: {total_specific}")
    print(f"Unified feature identities: {total_unified}")

    print("\nALL FEATURE SEPARATION CHECKS PASSED")
    print("No original input features were discarded.")

    print("=" * 65)


if __name__ == "__main__":
    main()
