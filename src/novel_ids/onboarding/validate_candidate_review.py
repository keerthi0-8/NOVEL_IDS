
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"

REVIEW_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_match_review.yaml"
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_yaml(CONFIG_PATH)
    review = load_yaml(REVIEW_PATH)

    # Load NSL-KDD metadata feature names.
    nsl_metadata = pd.read_csv(
        config["datasets"]["nsl_kdd"]["metadata"],
        header=None,
    )

    nsl_features = set(
        nsl_metadata.iloc[:, 0].astype(str).str.strip()
    )

    # Load actual UNSW-NB15 feature names.
    unsw_data = pd.read_csv(
        config["datasets"]["unsw_nb15"]["train"],
        nrows=0,
    )

    excluded = {"id", "attack_cat", "label"}

    unsw_features = {
        str(column)
        for column in unsw_data.columns
        if column not in excluded
    }

    candidates = review.get("candidates", [])
    errors = []

    allowed_statuses = {
        "pending_review",
        "not_approved",
        "rejected",
    }

    print("\n" + "=" * 60)
    print("NOVEL-IDS CANDIDATE REVIEW VALIDATION")
    print("=" * 60)

    for candidate in candidates:
        nsl_name = candidate["nsl_kdd"]
        unsw_names = candidate["unsw_nb15"]
        status = candidate["status"]

        nsl_ok = nsl_name in nsl_features
        unsw_ok = all(
            name in unsw_features for name in unsw_names
        )
        status_ok = status in allowed_statuses

        passed = nsl_ok and unsw_ok and status_ok

        print(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{nsl_name} -> {', '.join(unsw_names)}"
        )

        if not nsl_ok:
            errors.append(f"Missing NSL-KDD feature: {nsl_name}")

        for name in unsw_names:
            if name not in unsw_features:
                errors.append(f"Missing UNSW-NB15 feature: {name}")

        if not status_ok:
            errors.append(f"Invalid review status: {status}")

    print("\n" + "=" * 60)
    print(f"Candidates inspected: {len(candidates)}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
    else:
        print("ALL CANDIDATE REFERENCES PASSED")
        print("No candidate was added to the approved registry.")

    print("=" * 60)


if __name__ == "__main__":
    main()
