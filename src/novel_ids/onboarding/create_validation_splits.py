
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "validation_splits"

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD ORIGINAL TRAINING DATA ONLY
# --------------------------------------------------

def load_training_data(dataset_name, dataset_config):

    train_path = Path(dataset_config["train"])

    if dataset_name == "nsl_kdd":
        df = pd.read_csv(train_path, header=None)

        # NSL-KDD:
        # Columns 0-40 = input features
        # Column 41 = attack label
        # Column 42 = difficulty (excluded)

        X = df.iloc[:, :41].copy()
        y = df.iloc[:, 41].astype(str).copy()

    elif dataset_name == "unsw_nb15":
        df = pd.read_csv(train_path)

        # Exclude identifiers and target columns.
        excluded = ["id", "attack_cat", "label"]

        X = df.drop(
            columns=excluded,
            errors="ignore"
        ).copy()

        y = df["attack_cat"].astype(str).copy()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y


# --------------------------------------------------
# CREATE LEAKAGE-SAFE GROUPS
# --------------------------------------------------

def create_groups(X, y, dataset_name):

    # Identical feature records must stay together,
    # even if their labels differ.

    feature_hashes = pd.util.hash_pandas_object(
        X,
        index=False
    )

    groups = feature_hashes.to_numpy()

    group_labels = pd.DataFrame({
        "group": groups,
        "label": y.to_numpy()
    })

    # A feature-identical group with conflicting labels
    # needs investigation before splitting.

    conflicts = (
        group_labels.groupby("group")["label"]
        .nunique()
    )

    conflicting_groups = int((conflicts > 1).sum())

    print("Feature-identical groups with conflicting labels:",
          conflicting_groups)

    if conflicting_groups:
        print("\nCONFLICTING FEATURE GROUP DETAILS")

    conflicting_hashes = conflicts[conflicts > 1].index

    for group_hash in conflicting_hashes:
        mask = groups == group_hash

        print("\nGroup hash:", group_hash)
        print("Number of records:", int(mask.sum()))

        print(
            "Labels and counts:",
            y.iloc[np.where(mask)[0]].value_counts().to_dict()
        )

        print(
            "Representative features:",
            X.iloc[np.where(mask)[0][0]].head(10).to_dict()
        )

    return groups


# --------------------------------------------------
# CREATE STRATIFIED 80:20 SPLIT
# --------------------------------------------------

def create_split(dataset_name, dataset_config):

    output_path = OUTPUT_DIR / f"{dataset_name}_split.joblib"
    if output_path.exists():
        raise FileExistsError(
            f"Baseline split already exists and is protected from overwrite: {output_path}"
        )

    print("\n" + "=" * 65)
    print(f"{dataset_name.upper()} VALIDATION SPLIT")
    print("=" * 65)

    X, y = load_training_data(
        dataset_name,
        dataset_config
    )

    print("Original training records:", len(X))
    print("Input features:", X.shape[1])
    print("Number of classes:", y.nunique())

    class_counts = y.value_counts()

    print("\nClass counts:")
    print(class_counts.to_string())

    groups = create_groups(X, y, dataset_name)

    unique_groups = len(np.unique(groups))

    print("\nUnique feature groups:", unique_groups)

    if unique_groups < N_SPLITS:
        raise ValueError(
            "Not enough unique feature groups for a five-fold split."
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_classes = set(y.unique())
    candidates = []

    for train_idx, val_idx in splitter.split(
        X,
        y,
        groups
    ):

        train_classes = set(y.iloc[train_idx].unique())

        # Every original training class must remain
        # represented in the new training partition.

        if train_classes != all_classes:
            continue

        validation_fraction = len(val_idx) / len(X)

        score = abs(validation_fraction - 0.20)

        candidates.append(
            (score, train_idx, val_idx)
        )

    if not candidates:
        raise ValueError(
            "Could not create a split that preserves every class "
            "in training. Rare classes require review."
        )

    candidates.sort(key=lambda item: item[0])

    _, train_idx, val_idx = candidates[0]

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    X_val = X.iloc[val_idx].copy()
    y_val = y.iloc[val_idx].copy()

    # Verify no feature-identical groups cross partitions.

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise AssertionError(
            "Feature-identical groups crossed the split."
        )

    print("\nFinal split:")
    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print(
        "Validation percentage:",
        round(len(X_val) / len(X) * 100, 2),
        "%"
    )

    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())

    print("\nValidation class counts:")
    print(y_val.value_counts().to_string())

    missing_validation_classes = (
        all_classes - set(y_val.unique())
    )

    if missing_validation_classes:
        print("\nClasses absent from validation:")
        print(sorted(missing_validation_classes))
        print(
            "These classes remain in training. "
            "Their validation performance cannot be measured "
            "from this split."
        )

    # Save without modifying original dataset files.

    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_val,
            "y_validation": y_val,
            "random_state": RANDOM_STATE,
            "validation_fraction": 0.20,
            "grouped_by_feature_hash": True,
        },
        output_path
    )

    print("\nSaved:", output_path.relative_to(PROJECT_ROOT))
    print("PASS: No identical feature groups cross partitions.")

    return output_path


# --------------------------------------------------
# RUN BOTH SOURCE DATASETS
# --------------------------------------------------

if __name__ == "__main__":

    for dataset_name in ["nsl_kdd", "unsw_nb15"]:

        create_split(
            dataset_name,
            config["datasets"][dataset_name]
        )

    print("\nVALIDATION SPLIT GENERATION COMPLETED.")
