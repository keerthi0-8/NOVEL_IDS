
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from novel_ids.onboarding.load_source_datasets import load_config, validation_split_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DESCRIPTOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_descriptors.yaml"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "validation_splits"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_statistics.yaml"
)


def calculate_statistics(series):
    """Calculate feature statistics from source training data only."""

    missing_mask = series.isna()

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

        missing_fraction = float(numeric.isna().mean())
        valid = numeric.dropna()

        if valid.empty:
            return {
                "q01": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q99": None,
                "mean": None,
                "std": None,
                "skewness": None,
                "zero_fraction": None,
                "distinct_count": 0,
                "missing_fraction": missing_fraction,
                "is_binary": False,
                "is_integer": False,
            }

        distinct_count = int(valid.nunique())

        is_integer = bool(
            np.isclose(
                valid.to_numpy(dtype=np.float64),
                np.round(valid.to_numpy(dtype=np.float64)),
            ).all()
        )

        return {
            "q01": float(valid.quantile(0.01)),
            "q25": float(valid.quantile(0.25)),
            "median": float(valid.median()),
            "q75": float(valid.quantile(0.75)),
            "q99": float(valid.quantile(0.99)),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "skewness": (
                float(valid.skew())
                if distinct_count > 2
                else 0.0
            ),
            "zero_fraction": float((valid == 0).mean()),
            "distinct_count": distinct_count,
            "missing_fraction": missing_fraction,
            "is_binary": distinct_count <= 2,
            "is_integer": is_integer,
        }

    valid = series.dropna()
    distinct_count = int(valid.nunique())

    return {
        "q01": None,
        "q25": None,
        "median": None,
        "q75": None,
        "q99": None,
        "mean": None,
        "std": None,
        "skewness": None,
        "zero_fraction": None,
        "distinct_count": distinct_count,
        "missing_fraction": float(missing_mask.mean()),
        "is_binary": distinct_count <= 2,
        "is_integer": False,
    }


def main():
    print("=" * 65)
    print("NOVEL-IDS SOURCE TRAINING FEATURE STATISTICS")
    print("=" * 65)

    with open(DESCRIPTOR_PATH, encoding="utf-8") as file:
        descriptor_data = yaml.safe_load(file)

    descriptors = descriptor_data["features"]

    config = load_config()
    splits = {
        name: joblib.load(validation_split_path(config, name))
        for name in ("nsl_kdd", "unsw_nb15")
    }

    statistics_output = {
        "schema_version": 1,
        "fitted_on": "source_training_partitions_only",
        "datasets": {},
    }

    total_features = 0

    for dataset_name, split in splits.items():
        X_train = split["X_train"]

        dataset_descriptors = [
            item
            for item in descriptors
            if item["dataset"] == dataset_name
        ]

        if len(dataset_descriptors) != X_train.shape[1]:
            raise ValueError(
                f"{dataset_name}: descriptor count "
                f"({len(dataset_descriptors)}) does not match "
                f"training feature count ({X_train.shape[1]})."
            )

        dataset_statistics = {}

        for descriptor in dataset_descriptors:
            feature_name = descriptor["original_feature_name"]

            if dataset_name == "nsl_kdd":
                # NSL-KDD X_train uses integer column indices.
                # Descriptor order follows the original NSL feature order.
                column_index = dataset_descriptors.index(descriptor)
                series = X_train.iloc[:, column_index]

            else:
                if feature_name not in X_train.columns:
                    raise ValueError(
                        f"UNSW feature not found: {feature_name}"
                    )

                series = X_train[feature_name]

            dataset_statistics[feature_name] = (
                calculate_statistics(series)
            )

        statistics_output["datasets"][dataset_name] = (
            dataset_statistics
        )

        total_features += len(dataset_statistics)

        print(f"\n{dataset_name.upper()}")
        print("Training records:", len(X_train))
        print("Features processed:", len(dataset_statistics))
        print("Statistics source: X_train only")

    if total_features != 83:
        raise ValueError(
            f"Expected 83 feature entries, got {total_features}."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            statistics_output,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\nTotal feature entries:", total_features)
    print("Statistics saved to:", OUTPUT_PATH)
    print("\nFEATURE STATISTICS GENERATION PASSED.")


if __name__ == "__main__":
    main()
