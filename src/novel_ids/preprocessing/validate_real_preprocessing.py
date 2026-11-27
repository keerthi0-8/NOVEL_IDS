
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset_preprocessor import DatasetPreprocessor
from novel_ids.onboarding.load_source_datasets import load_config, validation_split_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT / "data" / "interim" / "validation_splits"
)



def choose_log_features(dataset_name, X_train):
    """Select justified byte-count transformations using training data only."""

    candidates = {
        "nsl_kdd": {
            4: "src_bytes",
            5: "dst_bytes",
        },
        "unsw_nb15": {
            "sbytes": "sbytes",
            "dbytes": "dbytes",
        },
    }

    selected = []

    print("\nByte-feature transformation assessment:")

    for column, feature_name in candidates[dataset_name].items():

        if column not in X_train.columns:
            print(f"{feature_name}: column not found; not selected")
            continue

        values = pd.to_numeric(
            X_train[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            print(f"{feature_name}: no valid values; not selected")
            continue

        skewness = values.skew()
        non_negative = bool((values >= 0).all())

        use_log = (
            non_negative
            and np.isfinite(skewness)
            and skewness > 1.0
        )

        print(
            f"{feature_name}: skewness={skewness:.3f}, "
            f"non_negative={non_negative}, "
            f"log1p_selected={use_log}"
        )

        if use_log:
            selected.append(column)

    return selected


def validate_dataset(name, config):
    split_path = validation_split_path(config, name)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Validation split not found: {split_path}"
        )

    data = joblib.load(split_path)

    X_train = data["X_train"]
    X_val = data["X_validation"]

    # Preserve the original frames for integrity checks.
    original_train = X_train.copy(deep=True)
    original_val = X_val.copy(deep=True)

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    log_features = choose_log_features(name, X_train)

    processor = DatasetPreprocessor(
        log_features=log_features
    )

    train_result = processor.fit_transform(X_train)
    val_result = processor.transform(X_val)

    numeric_train = train_result["numeric_values"].to_numpy()
    numeric_val = val_result["numeric_values"].to_numpy()

    assert np.isfinite(numeric_train).all()
    assert np.isfinite(numeric_val).all()

    assert X_train.equals(original_train)
    assert X_val.equals(original_val)

    assert len(train_result["numeric_values"]) == len(X_train)
    assert len(val_result["numeric_values"]) == len(X_val)

    assert len(train_result["categorical_values"]) == len(X_train)
    assert len(val_result["categorical_values"]) == len(X_val)

    assert (
        train_result["numeric_missing_mask"].shape
        == train_result["numeric_values"].shape
    )

    assert (
        val_result["numeric_missing_mask"].shape
        == val_result["numeric_values"].shape
    )

    assert (
        train_result["categorical_missing_mask"].shape
        == train_result["categorical_values"].shape
    )

    assert (
        val_result["categorical_missing_mask"].shape
        == val_result["categorical_values"].shape
    )

    print("\nNumeric features:", len(processor.numeric_columns))
    print("Categorical features:", len(processor.categorical_columns))
    print("Selected log1p features:", log_features)

    print(
        "Training numeric missing measurements:",
        int(train_result["numeric_missing_mask"].to_numpy().sum())
    )

    print(
        "Validation numeric missing measurements:",
        int(val_result["numeric_missing_mask"].to_numpy().sum())
    )

    print(
        "Training categorical missing measurements:",
        int(train_result["categorical_missing_mask"].to_numpy().sum())
    )

    print(
        "Validation categorical missing measurements:",
        int(val_result["categorical_missing_mask"].to_numpy().sum())
    )

    if processor.categorical_processor is not None:
        vocabularies = processor.categorical_processor.vocabularies

        print(
            "Training category vocabulary sizes:",
            {
                feature: len(vocabulary)
                for feature, vocabulary in vocabularies.items()
            }
        )

    print("\nAll numeric outputs finite: PASS")
    print("Original input data unchanged: PASS")
    print("Training-only preprocessing: PASS")
    print("DATASET PREPROCESSING VALIDATION PASSED")


def main():
    config = load_config()
    for name in ["nsl_kdd", "unsw_nb15"]:
        validate_dataset(name, config)

    print("\nALL REAL-DATA PREPROCESSING CHECKS PASSED.")


if __name__ == "__main__":
    main()
