
from pathlib import Path
import pandas as pd

print("=" * 70)
print("NOVEL-IDS: SOURCE DATASET PARTITION VERIFICATION")
print("=" * 70)

DATASETS = {
    "NSL-KDD": {
        "train": Path(
            "G:/My Drive/NSL-KDD/NSL_KDD-master/KDDTrain+.txt"
        ),
        "test": Path(
            "G:/My Drive/NSL-KDD/NSL_KDD-master/KDDTest+.txt"
        ),
        "header": None,
        "expected_train": 125973,
        "expected_test": 22544,
    },
    "UNSW-NB15": {
        "train": Path(
            "G:/My Drive/UNSW-NB15/UNSW_NB15_training-set.csv"
        ),
        "test": Path(
            "G:/My Drive/UNSW-NB15/UNSW_NB15_testing-set.csv"
        ),
        "header": 0,
        "expected_train": 175341,
        "expected_test": 82332,
    },
}

for name, config in DATASETS.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    train_path = config["train"]
    test_path = config["test"]

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    train = pd.read_csv(
        train_path,
        header=config["header"]
    )

    test = pd.read_csv(
        test_path,
        header=config["header"]
    )

    print("\nTRAINING FILE")
    print("Path:", train_path)
    print("Rows:", len(train))
    print("Columns:", len(train.columns))

    print("\nTESTING FILE")
    print("Path:", test_path)
    print("Rows:", len(test))
    print("Columns:", len(test.columns))

    print("\nEXPECTED PARTITION COUNTS")
    print("Expected training rows:", config["expected_train"])
    print("Expected testing rows:", config["expected_test"])

    print("\nPARTITION COUNT CHECK")
    if (
        len(train) == config["expected_train"]
        and len(test) == config["expected_test"]
    ):
        print("PASS: Expected partition counts match.")
    else:
        print("REVIEW REQUIRED: Counts differ from expected convention.")

    print("\nFEATURE/COLUMN INSPECTION")
    if name == "NSL-KDD":
        print("First five training columns:")
        print(train.columns[:5].tolist())

        print("Total raw columns:", len(train.columns))
        print("Expected structure: 41 features + attack label + difficulty")

        print("\nAttack label distribution:")
        print(train.iloc[:, 41].value_counts().to_string())

        print("\nTest attack label distribution:")
        print(test.iloc[:, 41].value_counts().to_string())

    else:
        print("Training columns:")
        print(train.columns.tolist())

        if "label" in train.columns:
            print("\nTraining binary label distribution:")
            print(train["label"].value_counts().to_string())

        if "attack_cat" in train.columns:
            print("\nTraining attack category distribution:")
            print(train["attack_cat"].value_counts().to_string())

        if "label" in test.columns:
            print("\nTesting binary label distribution:")
            print(test["label"].value_counts().to_string())

        if "attack_cat" in test.columns:
            print("\nTesting attack category distribution:")
            print(test["attack_cat"].value_counts().to_string())

print("\n" + "=" * 70)
print("BOTH DATASET PARTITION INSPECTIONS COMPLETED")
print("No original files were modified.")
print("=" * 70)
