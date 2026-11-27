
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_ids.onboarding.load_source_datasets import (
    load_config,
    load_nsl_kdd,
    load_unsw_nb15,
)


def audit_dataset(name, X_train, y_train, X_test, y_test):

    print("\n" + "=" * 70)
    print(f"{name.upper()} FEATURE OVERLAP AND LABEL AUDIT")
    print("=" * 70)

    train_hashes = pd.util.hash_pandas_object(
        X_train,
        index=False,
    )

    test_hashes = pd.util.hash_pandas_object(
        X_test,
        index=False,
    )

    train_labels = y_train.astype(str).reset_index(drop=True)
    test_labels = y_test.astype(str).reset_index(drop=True)

    # Map every training feature hash to its observed labels.
    train_hash_labels = {}

    for feature_hash, label in zip(train_hashes, train_labels):
        train_hash_labels.setdefault(
            feature_hash,
            set(),
        ).add(label)

    # Identify test rows whose feature vectors occur in training.
    matching_hashes = test_hashes.isin(
        set(train_hash_labels)
    ).to_numpy()

    same_label = []
    conflicting_label = []

    for feature_hash, test_label, overlaps in zip(
        test_hashes,
        test_labels,
        matching_hashes,
    ):

        if not overlaps:
            same_label.append(False)
            conflicting_label.append(False)
            continue

        training_labels = train_hash_labels[feature_hash]

        if test_label in training_labels:
            same_label.append(True)
            conflicting_label.append(False)
        else:
            same_label.append(False)
            conflicting_label.append(True)

    same_label = pd.Series(same_label)
    conflicting_label = pd.Series(conflicting_label)

    overlapping_test_rows = int(matching_hashes.sum())

    unique_overlapping_hashes = len(
        set(test_hashes[matching_hashes])
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print(
        "Test rows with training feature matches:",
        overlapping_test_rows,
    )

    print(
        "Distinct overlapping feature vectors:",
        unique_overlapping_hashes,
    )

    print(
        "Overlapping rows with matching labels:",
        int(same_label.sum()),
    )

    print(
        "Overlapping rows with conflicting labels:",
        int(conflicting_label.sum()),
    )

    print(
        "Independent test rows:",
        int((~matching_hashes).sum()),
    )

    # Show a few examples of conflicting labels.
    if conflicting_label.any():

        print("\nExamples of conflicting labels:")

        conflict_indices = conflicting_label[
            conflicting_label
        ].index[:10]

        for index in conflict_indices:

            feature_hash = test_hashes.iloc[index]

            print(
                f"Test row {index}: "
                f"test label={test_labels.iloc[index]!r}, "
                f"training labels="
                f"{sorted(train_hash_labels[feature_hash])}"
            )

    else:
        print("\nNo conflicting labels found among overlapping rows.")

    print("=" * 70)


def main():

    config = load_config()

    (
        nsl_X_train,
        nsl_y_train,
        nsl_X_test,
        nsl_y_test,
    ) = load_nsl_kdd(config)

    (
        unsw_X_train,
        unsw_y_train,
        unsw_X_test,
        unsw_y_test,
    ) = load_unsw_nb15(config)

    audit_dataset(
        "NSL-KDD",
        nsl_X_train,
        nsl_y_train,
        nsl_X_test,
        nsl_y_test,
    )

    audit_dataset(
        "UNSW-NB15",
        unsw_X_train,
        unsw_y_train,
        unsw_X_test,
        unsw_y_test,
    )


if __name__ == "__main__":
    main()
