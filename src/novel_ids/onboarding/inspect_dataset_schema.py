
from pathlib import Path

import pandas as pd


NSL_DIR = Path(r"G:\My Drive\NSL-KDD\NSL_KDD-master")
UNSW_DIR = Path(r"G:\My Drive\UNSW-NB15")


def inspect_nsl_kdd():
    print("\n" + "=" * 65)
    print("NSL-KDD FEATURE AND LABEL INSPECTION")
    print("=" * 65)

    train = pd.read_csv(NSL_DIR / "KDDTrain+.txt", header=None)
    test = pd.read_csv(NSL_DIR / "KDDTest+.txt", header=None)

    print("\nTotal columns:", train.shape[1])
    print("Feature columns: 0 to 40")
    print("Attack label column: 41")
    print("Difficulty column: 42")

    print("\nTraining attack categories:")
    print(train[41].value_counts().to_string())

    print("\nTesting attack categories:")
    print(test[41].value_counts().to_string())

    print("\nFeature metadata preview:")
    metadata = pd.read_csv(NSL_DIR / "Field Names.csv")
    print(metadata.to_string(index=False))


def inspect_unsw_nb15():
    print("\n" + "=" * 65)
    print("UNSW-NB15 FEATURE AND LABEL INSPECTION")
    print("=" * 65)

    train = pd.read_csv(UNSW_DIR / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(UNSW_DIR / "UNSW_NB15_testing-set.csv")

    excluded = {"id", "attack_cat", "label"}
    features = [c for c in train.columns if c not in excluded]

    print("\nTotal columns:", len(train.columns))
    print("Candidate input features:", len(features))
    print("Excluded columns:", sorted(excluded))

    print("\nTraining attack categories:")
    print(train["attack_cat"].value_counts(dropna=False).to_string())

    print("\nTesting attack categories:")
    print(test["attack_cat"].value_counts(dropna=False).to_string())

    print("\nTraining binary label counts:")
    print(train["label"].value_counts(dropna=False).to_string())

    print("\nFeature data types:")
    print(train[features].dtypes.to_string())

    print("\nUNSW feature metadata preview:")
    metadata = pd.read_csv(
        UNSW_DIR / "NUSW-NB15_features.csv",
        encoding="cp1252"
    )
    print(metadata.to_string(index=False))


def main():
    inspect_nsl_kdd()
    inspect_unsw_nb15()


if __name__ == "__main__":
    main()
