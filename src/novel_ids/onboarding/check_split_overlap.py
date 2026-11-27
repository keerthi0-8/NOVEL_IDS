
from pathlib import Path

import pandas as pd


DATASETS = {
    "NSL-KDD": {
        "train": r"G:\My Drive\NSL-KDD\NSL_KDD-master\KDDTrain+.txt",
        "test": r"G:\My Drive\NSL-KDD\NSL_KDD-master\KDDTest+.txt",
        "header": None,
    },
    "unsw_nb15": {
    "train": Path(
        "G:/My Drive/UNSW-NB15/UNSW_NB15_testing-set.csv"
    ),
    "test": Path(
        "G:/My Drive/UNSW-NB15/UNSW_NB15_training-set.csv"
    ),
    "header": 0,
},
}



def check_overlap(dataset_name, config):

    print("\n" + "=" * 65)
    print(dataset_name.upper(), "TRAIN/TEST OVERLAP")
    print("=" * 65)

    train = pd.read_csv(
        config["train"],
        header=config["header"]
    )

    test = pd.read_csv(
        config["test"],
        header=config["header"]
    )

    train_hashes = pd.util.hash_pandas_object(
        train, index=False
    )

    test_hashes = pd.util.hash_pandas_object(
        test, index=False
    )

    train_hash_set = set(train_hashes)

    overlapping_test_rows = test_hashes.isin(
        train_hash_set
    )

    shared_unique_hashes = (
        set(train_hashes) & set(test_hashes)
    )

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Shared unique row hashes:", len(shared_unique_hashes))
    print("Overlapping test records:", int(overlapping_test_rows.sum()))
    print(
        "Independent test records:",
        int((~overlapping_test_rows).sum())
    )



def main():
    for name, settings in DATASETS.items():
        check_overlap(name, settings)


if __name__ == "__main__":
    main()
