
import pandas as pd
import joblib
from novel_ids.onboarding.exact_feature_matching import exact_feature_key, exact_feature_overlap_groups

from novel_ids.onboarding.load_source_datasets import (
    load_config,
    load_nsl_kdd,
    load_unsw_nb15,
    validation_split_path,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = PROJECT_ROOT / "data" / "interim" / "validation_splits"


def audit(name, X_train, X_validation, X_test):
    print("\n" + "=" * 65)
    print(f"{name.upper()} COMBINED DEVELOPMENT / TEST OVERLAP")
    print("=" * 65)

    if not (
        X_train.shape[1]
        == X_validation.shape[1]
        == X_test.shape[1]
    ):
        raise ValueError(f"{name}: feature counts do not match")

    # Standardize column labels without changing feature values.
    X_train = X_train.copy()
    X_validation = X_validation.copy()
    X_test = X_test.copy()

    X_train.columns = range(X_train.shape[1])
    X_validation.columns = range(X_validation.shape[1])
    X_test.columns = range(X_test.shape[1])

    train_groups = exact_feature_overlap_groups(X_test, X_train)
    val_groups = exact_feature_overlap_groups(X_test, X_validation)
    train_test_positions = {
        position for group in train_groups for position in group["left_positions"]
    }
    val_test_positions = {
        position for group in val_groups for position in group["left_positions"]
    }
    both_positions = train_test_positions & val_test_positions
    either_positions = train_test_positions | val_test_positions
    matched_keys = {
        exact_feature_key(X_test.iloc[position].tolist())
        for position in either_positions
    }

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Test rows:", len(X_test))
    print("Test rows matching training:", len(train_test_positions))
    print("Test rows matching validation:", len(val_test_positions))
    print("Test rows matching both:", len(both_positions))
    print("Test rows matching either:", len(either_positions))
    print("Test rows independent of both:", len(X_test) - len(either_positions))
    print("Distinct exact test feature vectors matching development:", len(matched_keys))


def main():
    config = load_config()

    nsl = load_nsl_kdd(config)
    unsw = load_unsw_nb15(config)

    nsl_split = joblib.load(validation_split_path(config, "nsl_kdd"))
    unsw_split = joblib.load(
        validation_split_path(config, "unsw_nb15")
    )

    audit(
        "NSL-KDD",
        nsl_split["X_train"],
        nsl_split["X_validation"],
        nsl[2],
    )

    audit(
        "UNSW-NB15",
        unsw_split["X_train"],
        unsw_split["X_validation"],
        unsw[2],
    )


if __name__ == "__main__":
    main()
