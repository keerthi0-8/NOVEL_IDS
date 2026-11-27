
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


def validation_split_path(config, dataset_name):
    """Resolve the configured active split, preserving the baseline fallback."""
    datasets = config.get("datasets", config)
    dataset = datasets[dataset_name]
    configured = dataset.get("validation_split")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data" / "interim" / "validation_splits" / f"{dataset_name}_split.joblib"


def unified_artifact_path(config, dataset_name):
    """Resolve a dataset's configured unified representation output."""
    datasets = config.get("datasets", config)
    dataset = datasets[dataset_name]
    configured = dataset.get("unified_artifact")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data" / "processed" / "unified_representation" / f"{dataset_name}_unified.joblib"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_nsl_kdd(config):
    paths = config["datasets"]["nsl_kdd"]

    # Read feature names from the NSL-KDD metadata.
    metadata = pd.read_csv(
        paths["metadata"],
        header=None,
    )

    feature_names = (
        metadata.iloc[:, 0].astype(str).str.strip().tolist()
    )

    if len(feature_names) != 41:
        raise ValueError(
            f"Expected 41 NSL-KDD feature names, "
            f"found {len(feature_names)}"
        )

    column_names = feature_names + ["attack_label", "difficulty"]

    train = pd.read_csv(
        paths["train"],
        header=None,
        names=column_names,
    )

    test = pd.read_csv(
        paths["test"],
        header=None,
        names=column_names,
    )

    X_train = train[feature_names].copy()
    y_train = train["attack_label"].copy()

    X_test = test[feature_names].copy()
    y_test = test["attack_label"].copy()

    return X_train, y_train, X_test, y_test


def load_unsw_nb15(config):
    paths = config["datasets"]["unsw_nb15"]

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])

    excluded_columns = {"id", "attack_cat", "label"}

    feature_names = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    if len(feature_names) != 42:
        raise ValueError(
            f"Expected 42 UNSW-NB15 input features, "
            f"found {len(feature_names)}"
        )

    X_train = train[feature_names].copy()
    y_train = train["attack_cat"].copy()

    X_test = test[feature_names].copy()
    y_test = test["attack_cat"].copy()

    return X_train, y_train, X_test, y_test


def report_dataset(name, X_train, y_train, X_test, y_test):
    print(f"\n{'=' * 60}")
    print(f"{name} DATASET LOADING REPORT")
    print("=" * 60)

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Input features: {X_train.shape[1]}")

    print(f"Training labels: {y_train.nunique()}")
    print(f"Testing labels: {y_test.nunique()}")

    print(f"Training missing values: {X_train.isna().sum().sum()}")
    print(f"Testing missing values: {X_test.isna().sum().sum()}")

    print(f"Training feature columns match test: "
          f"{list(X_train.columns) == list(X_test.columns)}")

    print("\nTraining attack-label distribution:")
    print(y_train.value_counts().to_string())

    print("\nInput feature preview:")
    print(X_train.head(3).to_string())


def main():
    config = load_config()

    print("\nNOVEL-IDS SOURCE DATASET LOADER")
    print("Loading original source datasets without preprocessing.")

    nsl = load_nsl_kdd(config)
    unsw = load_unsw_nb15(config)

    report_dataset("NSL-KDD", *nsl)
    report_dataset("UNSW-NB15", *unsw)

    print("\n" + "=" * 60)
    print("SOURCE DATASET LOADING COMPLETED")
    print("Original input features preserved.")
    print("Multi-class attack labels separated from input features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
