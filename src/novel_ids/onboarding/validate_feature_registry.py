
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"
REGISTRY_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "semantic_feature_registry.yaml"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    registry = load_registry()

    datasets = config["datasets"]
    common_features = registry["common_features"]

    # Read only the training data headers/features.
    nsl_columns = list(
        pd.read_csv(
            datasets["nsl_kdd"]["train"],
            header=None,
            nrows=0,
        ).columns
    )

    unsw_columns = list(
        pd.read_csv(
            datasets["unsw_nb15"]["train"],
            nrows=0,
        ).columns
    )

    # NSL-KDD uses positional column indices.
    nsl_feature_names = [
        str(i) for i in range(41)
    ]

    # UNSW-NB15 uses named columns.
    unsw_feature_names = [
        column
        for column in unsw_columns
        if column not in {"id", "attack_cat", "label"}
    ]

    print("\n" + "=" * 60)
    print("NOVEL-IDS SEMANTIC FEATURE REGISTRY VALIDATION")
    print("=" * 60)

    print(f"\nNSL-KDD input features: {len(nsl_feature_names)}")
    print(f"UNSW-NB15 input features: {len(unsw_feature_names)}")

    errors = []

    print("\nCOMMON FEATURE MAPPINGS")
    print("-" * 60)

    for feature in common_features:
        nsl_name = feature["nsl_kdd"]
        unsw_name = feature["unsw_nb15"]

        # NSL-KDD registry names are semantic names,
        # so verify against the known NSL-KDD metadata names.
        nsl_metadata = pd.read_csv(
            datasets["nsl_kdd"]["metadata"],
            header=None,
        )

        nsl_metadata_names = (
            nsl_metadata.iloc[:, 0]
            .astype(str)
            .str.strip()
            .tolist()
        )

        nsl_exists = nsl_name in nsl_metadata_names
        unsw_exists = unsw_name in unsw_feature_names

        status = "PASS" if nsl_exists and unsw_exists else "FAIL"

        print(
            f"{status}: {feature['semantic_id']} | "
            f"NSL={nsl_name} | UNSW={unsw_name}"
        )

        if not nsl_exists:
            errors.append(f"NSL-KDD feature missing: {nsl_name}")

        if not unsw_exists:
            errors.append(f"UNSW-NB15 feature missing: {unsw_name}")

    print("\n" + "=" * 60)

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
    else:
        print("ALL REGISTERED FEATURE MAPPINGS PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()
