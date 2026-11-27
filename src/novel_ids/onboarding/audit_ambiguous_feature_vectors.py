
from collections import defaultdict

import pandas as pd

from novel_ids.onboarding.load_source_datasets import (
    load_config,
    load_nsl_kdd,
    load_unsw_nb15,
)


def audit(name, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 65)
    print(f"{name}: AMBIGUOUS FEATURE-VECTOR AUDIT")
    print("=" * 65)

    train_hashes = pd.util.hash_pandas_object(
        X_train, index=False
    )
    test_hashes = pd.util.hash_pandas_object(
        X_test, index=False
    )

    label_map = defaultdict(set)

    for h, label in zip(train_hashes, y_train.astype(str)):
        label_map[h].add(label)

    ambiguous_hashes = {
        h for h, labels in label_map.items()
        if len(labels) > 1
    }

    test_labels = y_test.astype(str).reset_index(drop=True)

    ambiguous_test = test_hashes.isin(
        ambiguous_hashes
    ).to_numpy()

    matched_test = test_hashes.isin(
        set(label_map)
    ).to_numpy()

    print("Unique ambiguous training feature vectors:",
          len(ambiguous_hashes))
    print("Test rows matching ambiguous training vectors:",
          int(ambiguous_test.sum()))
    print("All test rows matching training features:",
          int(matched_test.sum()))

    if ambiguous_test.any():
        print("\nExamples of ambiguous training feature vectors:")

        shown = 0
        for i, h in enumerate(test_hashes):
            if h in ambiguous_hashes:
                print(
                    f"Test row {i}: "
                    f"test_label={test_labels.iloc[i]!r}, "
                    f"training_labels={sorted(label_map[h])}"
                )
                shown += 1
                if shown == 10:
                    break


def main():
    config = load_config()

    nsl = load_nsl_kdd(config)
    unsw = load_unsw_nb15(config)

    audit("NSL-KDD", *nsl)
    audit("UNSW-NB15", *unsw)


if __name__ == "__main__":
    main()
