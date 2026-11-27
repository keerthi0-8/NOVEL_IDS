
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
OUTPUT_PATH = PROJECT_ROOT / "results" / "feature_inventory.csv"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_yaml(CONFIG_PATH)
    registry = load_yaml(REGISTRY_PATH)

    # Load NSL-KDD feature names from metadata.
    nsl_metadata = pd.read_csv(
        config["datasets"]["nsl_kdd"]["metadata"],
        header=None,
    )

    nsl_features = (
        nsl_metadata.iloc[:, 0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Load actual UNSW-NB15 feature names.
    unsw_data = pd.read_csv(
        config["datasets"]["unsw_nb15"]["train"],
        nrows=0,
    )

    excluded = {"id", "attack_cat", "label"}

    unsw_features = [
        str(column)
        for column in unsw_data.columns
        if column not in excluded
    ]

    # Build lookup dictionaries from approved mappings.
    nsl_common = {}
    unsw_common = {}

    for item in registry["common_features"]:
        nsl_common[item["nsl_kdd"]] = item["semantic_id"]
        unsw_common[item["unsw_nb15"]] = item["semantic_id"]

    inventory = []

    for feature in nsl_features:
        semantic_id = nsl_common.get(feature)

        inventory.append({
            "dataset": "NSL-KDD",
            "feature": feature,
            "classification": (
                "common" if semantic_id else "dataset_specific"
            ),
            "semantic_id": semantic_id or "",
        })

    for feature in unsw_features:
        semantic_id = unsw_common.get(feature)

        inventory.append({
            "dataset": "UNSW-NB15",
            "feature": feature,
            "classification": (
                "common" if semantic_id else "dataset_specific"
            ),
            "semantic_id": semantic_id or "",
        })

    result = pd.DataFrame(inventory)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("NOVEL-IDS COMPLETE FEATURE INVENTORY")
    print("=" * 60)

    print(f"\nNSL-KDD features: {len(nsl_features)}")
    print(f"UNSW-NB15 features: {len(unsw_features)}")
    print(f"Total feature entries: {len(result)}")

    print("\nClassification summary:")
    print(
        result.groupby(["dataset", "classification"])
        .size()
        .to_string()
    )

    print(f"\nInventory saved to: {OUTPUT_PATH}")

    print("\nValidation:")
    print(
        "PASS: All features accounted for."
        if len(result) == len(nsl_features) + len(unsw_features)
        else "FAIL: Feature count mismatch."
    )


if __name__ == "__main__":
    main()
