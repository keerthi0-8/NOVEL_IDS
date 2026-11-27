
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

DESCRIPTOR_FILE = (
    ROOT / "src" / "novel_ids" / "onboarding"
    / "feature_descriptors.yaml"
)


def main():
    with open(DESCRIPTOR_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    features = data["features"]

    print("=" * 70)
    print("NOVEL-IDS THREE-LEVEL FEATURE AUDIT")
    print("=" * 70)

    print("\nTOTAL FEATURES:", len(features))
    print("DECLARED FEATURE COUNT:", data.get("feature_count"))

    for level in (1, 2, 3):
        records = [
            item for item in features
            if item.get("level") == level
        ]
        print(f"Level {level}: {len(records)}")

    print("\nLEVEL 1 VALIDATION")
    print("-" * 70)

    level1 = [f for f in features if f.get("level") == 1]

    level1_errors = []

    for f in level1:
        if not f.get("semantic_id"):
            level1_errors.append(f)

        if f.get("relationship") != "equivalent":
            level1_errors.append(f)

        if f.get("review_status") != "approved":
            level1_errors.append(f)

        if not f.get("evidence"):
            level1_errors.append(f)

    print("Level 1 records:", len(level1))
    print("Level 1 validation errors:", len(level1_errors))

    print("\nLEVEL 2 VALIDATION")
    print("-" * 70)

    level2 = [f for f in features if f.get("level") == 2]

    allowed_relationships = {"group_only", "partial"}
    level2_errors = []

    for f in level2:
        if not f.get("semantic_group"):
            level2_errors.append(f)

        if f.get("relationship") not in allowed_relationships:
            level2_errors.append(f)

        if not f.get("evidence"):
            level2_errors.append(f)

        if f.get("confidence") not in {"high", "medium", "low"}:
            level2_errors.append(f)

    print("Level 2 records:", len(level2))
    print("Level 2 validation errors:", len(level2_errors))

    print("\nLEVEL 3 VALIDATION")
    print("-" * 70)

    level3 = [f for f in features if f.get("level") == 3]

    level3_errors = []

    for f in level3:
        if f.get("semantic_id") is not None:
            level3_errors.append(f)

        if f.get("semantic_group") is not None:
            level3_errors.append(f)

        if f.get("relationship") != "dataset_specific":
            level3_errors.append(f)

    print("Level 3 records:", len(level3))
    print("Level 3 validation errors:", len(level3_errors))

    print("\nDESCRIPTION AND UNIT REVIEW")
    print("-" * 70)

    low_description = [
        f for f in features
        if f.get("description_confidence") == "low"
    ]

    unknown_units = [
        f for f in features
        if f.get("unit_status") == "unknown_or_not_provided"
    ]

    print("Low-confidence descriptions:", len(low_description))
    print("Unknown or unprovided units:", len(unknown_units))

    print("\nLEVEL 1 FEATURE PAIRING")
    print("-" * 70)

    semantic_ids = {}

    for f in level1:
        semantic_ids.setdefault(
            f["semantic_id"], []
        ).append(
            (f["dataset"], f["original_feature_name"])
        )

    for semantic_id, members in semantic_ids.items():
        print(semantic_id, "->", members)

    print("\nFINAL AUDIT")
    print("-" * 70)

    errors = (
        len(level1_errors)
        + len(level2_errors)
        + len(level3_errors)
    )

    print("Total structural validation errors:", errors)

    if errors == 0:
        print("PASS: Three-level structural rules satisfied.")
    else:
        print("REVIEW REQUIRED: Structural issues found.")

    print("No descriptor or registry files were modified.")


if __name__ == "__main__":
    main()
