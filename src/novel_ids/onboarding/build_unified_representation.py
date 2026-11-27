
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml


# =========================================================
# 1. PROJECT PATHS AND IMPORTS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_ids.onboarding.load_source_datasets import (
    load_config,
    load_nsl_kdd,
    load_unsw_nb15,
    unified_artifact_path,
    validation_split_path,
)

from novel_ids.preprocessing.dataset_preprocessor import (
    DatasetPreprocessor,
)

from novel_ids.representation.feature_identity import (
    FeatureIdentityEncoder,
)


SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "validation_splits"
)

DESCRIPTOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "novel_ids"
    / "onboarding"
    / "feature_descriptors_with_stats.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_representation"
)

RESULTS_DIR = PROJECT_ROOT / "results"

EXPECTED_SOURCE_IDENTITIES = 78
EXPECTED_NSL_FEATURES = 41
EXPECTED_UNSW_FEATURES = 42

RANDOM_SEED = 42


# =========================================================
# 2. FEATURE IDENTITY KEYS
# =========================================================

def get_token_key(descriptor):
    """
    Level 1:
        Use the shared canonical semantic identity.

    Level 2 and Level 3:
        Preserve individual dataset-specific identities.

    Related features are not merged merely because they
    belong to the same semantic group.
    """

    level = descriptor.get("level")
    semantic_id = descriptor.get("semantic_id")

    if level == 1 and semantic_id:
        return semantic_id

    return (
        f"{descriptor['dataset']}::"
        f"{descriptor['original_feature_name']}"
    )


def build_token_schema(descriptors):
    """
    Construct the 78 source identity slots.

    5 shared Level 1 identities.
    36 NSL-KDD-specific identities.
    37 UNSW-NB15-specific identities.
    """

    common_keys = []
    nsl_keys = []
    unsw_keys = []

    descriptor_by_key = {}

    for descriptor in descriptors:
        key = get_token_key(descriptor)

        descriptor_by_key.setdefault(key, descriptor)

        if descriptor.get("level") == 1:
            if key not in common_keys:
                common_keys.append(key)

        elif descriptor["dataset"] == "nsl_kdd":
            if key not in nsl_keys:
                nsl_keys.append(key)

        elif descriptor["dataset"] == "unsw_nb15":
            if key not in unsw_keys:
                unsw_keys.append(key)

        else:
            raise ValueError(
                "Unexpected source dataset in descriptor: "
                f"{descriptor['dataset']}"
            )

    token_keys = common_keys + nsl_keys + unsw_keys

    if len(common_keys) != 5:
        raise ValueError(
            f"Expected 5 common identities, got {len(common_keys)}"
        )

    if len(nsl_keys) != 36:
        raise ValueError(
            f"Expected 36 NSL-specific identities, got {len(nsl_keys)}"
        )

    if len(unsw_keys) != 37:
        raise ValueError(
            f"Expected 37 UNSW-specific identities, got {len(unsw_keys)}"
        )

    if len(token_keys) != EXPECTED_SOURCE_IDENTITIES:
        raise ValueError(
            f"Expected 78 identities, got {len(token_keys)}"
        )

    if len(set(token_keys)) != len(token_keys):
        raise ValueError("Duplicate token identities detected.")

    token_to_index = {
        key: index
        for index, key in enumerate(token_keys)
    }

    schema_features = []

    for index, key in enumerate(token_keys):
        descriptor = descriptor_by_key[key]

        schema_features.append({
            "index": index,
            "token_key": key,
            "level": descriptor.get("level"),
            "semantic_id": descriptor.get("semantic_id"),
            "semantic_group": descriptor.get("semantic_group"),
            "dataset": descriptor.get("dataset"),
            "original_feature_name": descriptor.get(
                "original_feature_name"
            ),
            "data_type": descriptor.get("data_type"),
            "relationship": descriptor.get("relationship"),
            "confidence": descriptor.get("confidence"),
        })

    return token_keys, token_to_index, schema_features


def build_dataset_feature_mapping(descriptors, dataset_name):
    """Map original feature names to token identities."""

    mapping = {}

    for descriptor in descriptors:
        if descriptor["dataset"] != dataset_name:
            continue

        feature = descriptor["original_feature_name"]
        key = get_token_key(descriptor)

        if feature in mapping:
            raise ValueError(
                f"Duplicate descriptor: {dataset_name}/{feature}"
            )

        mapping[feature] = key

    return mapping


# =========================================================
# 3. TRAINING-ONLY DESCRIPTOR STATISTICS
# =========================================================

def calculate_feature_statistics(series):
    """
    Calculate feature statistics using source training data only.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = numeric.dropna()

    missing_fraction = float(numeric.isna().mean())

    distinct_count = int(series.nunique(dropna=True))

    if valid.empty:
        return {
            "q01": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q99": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "zero_fraction": 0.0,
            "distinct_count": distinct_count,
            "missing_fraction": missing_fraction,
            "is_binary": distinct_count <= 2,
            "is_integer": False,
        }

    is_integer = bool(
        np.allclose(
            valid.to_numpy(),
            np.round(valid.to_numpy()),
        )
    )

    is_binary = valid.nunique() <= 2

    skewness = valid.skew()

    if not np.isfinite(skewness):
        skewness = 0.0

    return {
        "q01": float(valid.quantile(0.01)),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "q75": float(valid.quantile(0.75)),
        "q99": float(valid.quantile(0.99)),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "skewness": float(skewness),
        "zero_fraction": float((valid == 0).mean()),
        "distinct_count": distinct_count,
        "missing_fraction": missing_fraction,
        "is_binary": bool(is_binary),
        "is_integer": is_integer,
    }


def refresh_descriptor_statistics(descriptors, training_data):
    """
    Refresh descriptor statistics using only the audited
    source training partitions.
    """

    refreshed = []

    for descriptor in descriptors:
        updated = dict(descriptor)

        dataset_name = updated["dataset"]
        feature_name = updated["original_feature_name"]

        if dataset_name not in training_data:
            raise ValueError(
                f"Missing training data for {dataset_name}"
            )

        X_train = training_data[dataset_name]

        if feature_name not in X_train.columns:
            raise ValueError(
                f"Descriptor feature not found: "
                f"{dataset_name}/{feature_name}"
            )

        updated["statistics"] = calculate_feature_statistics(
            X_train[feature_name]
        )

        updated["statistics_source"] = (
            "audited_source_training_partition_only"
        )

        refreshed.append(updated)

    return refreshed


# =========================================================
# 4. PREPROCESSING CONFIGURATION
# =========================================================

def choose_log_features(dataset_name, X_train):
    """
    Select non-negative, skewed byte-count features.
    Uses training data only.
    """

    candidates = {
        "nsl_kdd": [
            "src_bytes",
            "dst_bytes",
        ],
        "unsw_nb15": [
            "sbytes",
            "dbytes",
        ],
    }

    if dataset_name not in candidates:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    selected = []

    for feature in candidates[dataset_name]:

        if feature not in X_train.columns:
            continue

        values = pd.to_numeric(
            X_train[feature],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if values.empty:
            continue

        non_negative = bool((values >= 0).all())
        skewness = float(values.skew())

        if (
            non_negative
            and np.isfinite(skewness)
            and skewness > 1.0
        ):
            selected.append(feature)

        print(
            f"{feature}: "
            f"skewness={skewness:.3f}, "
            f"non_negative={non_negative}, "
            f"log1p_selected={feature in selected}"
        )

    return selected


def preprocess_dataset(
    dataset_name,
    X_train,
    X_validation,
    X_test,
):
    """
    Fit preprocessing on training only.
    Transform validation and original test using fitted parameters.
    """

    print("\n" + "=" * 65)
    print(f"{dataset_name.upper()} PREPROCESSING")
    print("=" * 65)

    log_features = choose_log_features(
        dataset_name,
        X_train,
    )

    processor = DatasetPreprocessor(
        log_features=log_features,
    )

    processor.fit(X_train)

    train_processed = processor.transform(X_train)

    validation_processed = processor.transform(
        X_validation
    )

    test_processed = processor.transform(X_test)

    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Original test:", X_test.shape)
    print("Numeric features:", len(processor.numeric_columns))
    print("Categorical features:", len(processor.categorical_columns))
    print("Selected log1p features:", log_features)

    return (
        processor,
        train_processed,
        validation_processed,
        test_processed,
    )


# =========================================================
# 5. BUILD FEATURE TOKEN ARRAYS
# =========================================================

def create_token_data(
    processed,
    source_feature_names,
    token_keys,
    token_to_index,
    feature_mapping,
):
    """
    Create the 78-slot representation.

    Actual available measurements are preserved.

    NaN and -1 are storage placeholders only for structurally
    unavailable slots. They must not be passed directly
    into the neural network.

    M_avail:
        1 = feature exists in this dataset.
        0 = structurally unavailable.

    M_miss:
        1 = existing measurement was missing.
        0 = observed measurement.
    """

    numeric_frame = processed["numeric_values"]
    categorical_frame = processed["categorical_values"]

    row_count = len(numeric_frame) if not numeric_frame.empty else (
        len(categorical_frame)
    )

    if row_count == 0:
        raise ValueError("Cannot build tokens for an empty partition.")

    token_count = len(token_keys)

    numeric_values = np.full(
        (row_count, token_count),
        np.nan,
        dtype=np.float32,
    )

    categorical_values = np.full(
        (row_count, token_count),
        -1,
        dtype=np.int64,
    )

    M_avail = np.zeros(
        (row_count, token_count),
        dtype=np.float32,
    )

    M_miss = np.zeros(
        (row_count, token_count),
        dtype=np.float32,
    )

    numeric_token_mask = np.zeros(
        token_count,
        dtype=bool,
    )

    categorical_token_mask = np.zeros(
        token_count,
        dtype=bool,
    )

    if len(source_feature_names) != len(feature_mapping):
        raise ValueError(
            "Source feature count and feature mapping count differ."
        )

    for feature in source_feature_names:

        if feature not in feature_mapping:
            raise ValueError(
                f"Missing feature mapping: {feature}"
            )

        token_key = feature_mapping[feature]

        if token_key not in token_to_index:
            raise ValueError(
                f"Token identity not found: {token_key}"
            )

        index = token_to_index[token_key]

        M_avail[:, index] = 1.0

        if feature in processed["numeric_columns"]:

            numeric_values[:, index] = (
                processed["numeric_values"][feature]
                .to_numpy(dtype=np.float32)
            )

            M_miss[:, index] = (
                processed["numeric_missing_mask"][feature]
                .to_numpy(dtype=np.float32)
            )

            numeric_token_mask[index] = True

        elif feature in processed["categorical_columns"]:

            categorical_values[:, index] = (
                processed["categorical_values"][feature]
                .to_numpy(dtype=np.int64)
            )

            M_miss[:, index] = (
                processed["categorical_missing_mask"][feature]
                .to_numpy(dtype=np.float32)
            )

            categorical_token_mask[index] = True

        else:
            raise ValueError(
                f"Unrecognized processed feature: {feature}"
            )

    if not np.isin(M_avail, [0.0, 1.0]).all():
        raise ValueError("M_avail must be binary.")

    if not np.isin(M_miss, [0.0, 1.0]).all():
        raise ValueError("M_miss must be binary.")

    if np.any(M_miss > M_avail):
        raise ValueError(
            "Unavailable features cannot have measurement missingness."
        )

    available_numeric = numeric_values[:, numeric_token_mask]

    if (
        available_numeric.size
        and not np.isfinite(available_numeric).all()
    ):
        raise ValueError(
            "Available numeric measurements contain NaN or Inf."
        )

    available_categorical = categorical_values[
        :,
        categorical_token_mask,
    ]

    if (
        available_categorical.size
        and (available_categorical < 0).any()
    ):
        raise ValueError(
            "Available categorical measurements have invalid IDs."
        )

    return {
        "numeric_values": numeric_values,
        "categorical_values": categorical_values,
        "M_avail": M_avail,
        "M_miss": M_miss,
        "numeric_token_mask": numeric_token_mask,
        "categorical_token_mask": categorical_token_mask,
    }


# =========================================================
# 6. ORIGINAL TEST OVERLAP
# =========================================================

def independent_test_mask(X_original_train, X_original_test):
    """
    Exclude exact test feature-row matches found in the
    original source training file.

    Preserve both full and independent test partitions.
    """

    train_hashes = pd.util.hash_pandas_object(
        X_original_train,
        index=False,
    )

    test_hashes = pd.util.hash_pandas_object(
        X_original_test,
        index=False,
    )

    train_hash_set = set(train_hashes)

    overlap = test_hashes.isin(train_hash_set).to_numpy()

    return ~overlap


# =========================================================
# 7. BUILD TEST SUBSETS SAFELY
# =========================================================

def make_test_independent(test_tokens, labels, independent_mask):
    """
    Filter row-level arrays only.

    Feature-level arrays such as numeric_token_mask are
    deliberately not filtered.
    """

    row_keys = {
        "numeric_values",
        "categorical_values",
        "M_avail",
        "M_miss",
    }

    independent = {}

    for key, value in test_tokens.items():

        if key in row_keys:
            independent[key] = value[independent_mask]

        else:
            independent[key] = value.copy()

    independent["labels"] = (
        labels.iloc[independent_mask]
        .reset_index(drop=True)
    )

    return independent


def make_artifact(
    dataset_name,
    token_keys,
    source_feature_names,
    train_tokens,
    validation_tokens,
    test_tokens,
    y_train,
    y_validation,
    y_test,
    independent_mask,
    processor,
    identity_dimension,
):
    """Assemble all dataset partitions."""

    test_independent = make_test_independent(
        test_tokens,
        y_test,
        independent_mask,
    )

    return {
        "dataset": dataset_name,
        "token_keys": list(token_keys),
        "source_feature_names": list(source_feature_names),

        "train": {
            **train_tokens,
            "labels": y_train.reset_index(drop=True),
        },

        "validation": {
            **validation_tokens,
            "labels": y_validation.reset_index(drop=True),
        },

        "test_full": {
            **test_tokens,
            "labels": y_test.reset_index(drop=True),
            "independent_mask": independent_mask,
        },

        "test_independent": test_independent,

        "preprocessing": {
            "numeric_columns": list(processor.numeric_columns),
            "categorical_columns": list(processor.categorical_columns),
            "log_features": list(processor.log_features),
            "fit_on": "audited_source_training_partition_only",
        },

        "identity_dimension": identity_dimension,
        "random_seed": RANDOM_SEED,
    }


# =========================================================
# 8. IDENTITY ENCODER OUTPUT CONVERSION
# =========================================================

def to_numpy_identity(identity_tensor):
    """Convert identity output to CPU NumPy."""

    if isinstance(identity_tensor, torch.Tensor):
        return identity_tensor.detach().cpu().numpy()

    return np.asarray(identity_tensor)


# =========================================================
# 9. ARTIFACT VALIDATION
# =========================================================

def validate_artifact(artifact, expected_feature_count):
    """Validate saved partition dimensions and masks."""

    token_count = len(artifact["token_keys"])

    expected_available = expected_feature_count

    for split_name in [
        "train",
        "validation",
        "test_full",
        "test_independent",
    ]:

        split = artifact[split_name]

        row_count = len(split["labels"])

        for key in [
            "numeric_values",
            "categorical_values",
            "M_avail",
            "M_miss",
        ]:

            if split[key].shape != (row_count, token_count):
                raise ValueError(
                    f"{artifact['dataset']}/{split_name}: "
                    f"{key} has an unexpected shape."
                )

        if not np.isfinite(split["M_avail"]).all():
            raise ValueError("M_avail contains non-finite values.")

        if not np.isfinite(split["M_miss"]).all():
            raise ValueError("M_miss contains non-finite values.")

        if not np.isin(split["M_avail"], [0.0, 1.0]).all():
            raise ValueError("M_avail is not binary.")

        if not np.isin(split["M_miss"], [0.0, 1.0]).all():
            raise ValueError("M_miss is not binary.")

        if np.any(split["M_miss"] > split["M_avail"]):
            raise ValueError("M_miss exceeds M_avail.")

        if row_count:
            available_per_row = split["M_avail"].sum(axis=1)

            if not np.all(available_per_row == expected_available):
                raise ValueError(
                    f"{artifact['dataset']}/{split_name}: "
                    "unexpected number of available features."
                )

        numeric_mask = split["numeric_token_mask"]
        categorical_mask = split["categorical_token_mask"]

        if np.any(numeric_mask & categorical_mask):
            raise ValueError(
                "A token cannot be both numeric and categorical."
            )

        if int(numeric_mask.sum() + categorical_mask.sum()) != (
            expected_available
        ):
            raise ValueError(
                "Numeric and categorical token counts do not "
                "match the source feature count."
            )

        available_numeric = split["numeric_values"][:, numeric_mask]

        if (
            available_numeric.size
            and not np.isfinite(available_numeric).all()
        ):
            raise ValueError(
                "Available numeric values contain NaN or Inf."
            )

        available_categories = split["categorical_values"][
            :,
            categorical_mask,
        ]

        if (
            available_categories.size
            and (available_categories < 0).any()
        ):
            raise ValueError(
                "Available categorical values contain invalid IDs."
            )

    full_count = len(artifact["test_full"]["labels"])
    independent_count = len(artifact["test_independent"]["labels"])

    if independent_count > full_count:
        raise ValueError("Independent test count exceeds full test count.")

    return True


# =========================================================
# 10. MAIN PIPELINE
# =========================================================

def main():

    print("\n" + "=" * 70)
    print("NOVEL-IDS UNIFIED REPRESENTATION BUILDER")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # A. Load audited validation splits
    # -----------------------------------------------------

    print("\n[1/10] Loading audited validation splits...")

    config = load_config()
    nsl_split = joblib.load(validation_split_path(config, "nsl_kdd"))

    unsw_split = joblib.load(
        SPLIT_DIR / "unsw_nb15_split.joblib"
    )

    # -----------------------------------------------------
    # B. Load original train/test data
    # -----------------------------------------------------

    print("[2/10] Loading original source datasets...")

    (
        nsl_original_train,
        _,
        nsl_original_test,
        nsl_original_y_test,
    ) = load_nsl_kdd(config)

    (
        unsw_original_train,
        _,
        unsw_original_test,
        unsw_original_y_test,
    ) = load_unsw_nb15(config)

    
    # -----------------------------------------------------
    # C. Verify and normalize source partition columns
    # -----------------------------------------------------

    print("[3/10] Verifying source partitions...")

    # NSL-KDD validation splits were created using positional
    # columns (0 to 40). The original loader assigns actual
    # feature names from Field Names.csv.
    nsl_feature_names = list(nsl_original_train.columns)

    if len(nsl_split["X_train"].columns) != EXPECTED_NSL_FEATURES:
        raise ValueError(
            "NSL-KDD validation training split must contain "
            "exactly 41 features."
        )

    if len(nsl_split["X_validation"].columns) != EXPECTED_NSL_FEATURES:
        raise ValueError(
            "NSL-KDD validation partition must contain "
            "exactly 41 features."
        )

    # Rename by position only. The split was created from the
    # original NSL-KDD feature columns in their original order.
    nsl_split["X_train"].columns = nsl_feature_names
    nsl_split["X_validation"].columns = nsl_feature_names

    # Verify the corrected feature order.
    if list(nsl_split["X_train"].columns) != nsl_feature_names:
        raise ValueError(
            "NSL-KDD training feature order mismatch after normalization."
        )

    if list(nsl_split["X_validation"].columns) != nsl_feature_names:
        raise ValueError(
            "NSL-KDD validation feature order mismatch after normalization."
        )

    # UNSW-NB15 already uses named CSV columns.
    if list(unsw_split["X_train"].columns) != list(
        unsw_original_train.columns
    ):
        raise ValueError(
            "UNSW-NB15 training feature order mismatch."
        )

    if list(unsw_split["X_validation"].columns) != list(
        unsw_original_train.columns
    ):
        raise ValueError(
            "UNSW-NB15 validation feature order mismatch."
        )

    # Original train/test feature orders must also match.
    if list(nsl_original_train.columns) != list(
        nsl_original_test.columns
    ):
        raise ValueError(
            "NSL-KDD original train/test feature order mismatch."
        )

    if list(unsw_original_train.columns) != list(
        unsw_original_test.columns
    ):
        raise ValueError(
            "UNSW-NB15 original train/test feature order mismatch."
        )

    print("PASS: NSL-KDD feature columns normalized.")
    print("PASS: NSL-KDD feature order verified.")
    print("PASS: UNSW-NB15 feature order verified.")

    # -----------------------------------------------------
    # D. Load descriptors and refresh statistics
    # -----------------------------------------------------

    print("[3/10] Loading and refreshing descriptors...")

    if not DESCRIPTOR_PATH.exists():
        raise FileNotFoundError(
            f"Descriptor file not found: {DESCRIPTOR_PATH}"
        )

    with open(DESCRIPTOR_PATH, "r", encoding="utf-8") as file:
        descriptor_data = yaml.safe_load(file)

    original_descriptors = descriptor_data["features"]

    if len(original_descriptors) != 83:
        raise ValueError(
            "Expected 83 descriptors, found "
            f"{len(original_descriptors)}."
        )

    training_data = {
        "nsl_kdd": nsl_split["X_train"],
        "unsw_nb15": unsw_split["X_train"],
    }

    descriptors = refresh_descriptor_statistics(
        original_descriptors,
        training_data,
    )

    # Save exact descriptors used by this run.
    with open(
        OUTPUT_DIR / "training_feature_descriptors.yaml",
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            {
                "schema_version": 1,
                "statistics_source": (
                    "audited_source_training_partitions_only"
                ),
                "features": descriptors,
            },
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    # -----------------------------------------------------
    # E. Build token schema and feature mappings
    # -----------------------------------------------------

    print("[4/10] Building source token schema...")

    (
        token_keys,
        token_to_index,
        schema_features,
    ) = build_token_schema(descriptors)

    nsl_mapping = build_dataset_feature_mapping(
        descriptors,
        "nsl_kdd",
    )

    unsw_mapping = build_dataset_feature_mapping(
        descriptors,
        "unsw_nb15",
    )

    if len(nsl_mapping) != EXPECTED_NSL_FEATURES:
        raise ValueError(
            f"NSL-KDD expected 41 features, got {len(nsl_mapping)}"
        )

    if len(unsw_mapping) != EXPECTED_UNSW_FEATURES:
        raise ValueError(
            f"UNSW-NB15 expected 42 features, got {len(unsw_mapping)}"
        )

    print("Source identities:", len(token_keys))
    print("NSL-KDD features:", len(nsl_mapping))
    print("UNSW-NB15 features:", len(unsw_mapping))

    # -----------------------------------------------------
    # F. Fit feature identity encoder
    # -----------------------------------------------------

    print("[5/10] Generating feature identity vectors...")

    identity_encoder = FeatureIdentityEncoder()

    identity_encoder.fit(descriptors)

    identity_output = identity_encoder.encode(descriptors)

    descriptor_identities = to_numpy_identity(
        identity_output
    )

    if descriptor_identities.ndim != 2:
        raise ValueError(
            "Feature identities must be a 2D matrix."
        )

    if descriptor_identities.shape[0] != len(descriptors):
        raise ValueError(
            "Identity count does not match descriptor count."
        )

    identity_dimension = descriptor_identities.shape[1]

    identity_vectors = np.zeros(
        (len(token_keys), identity_dimension),
        dtype=np.float32,
    )

    identity_assigned = np.zeros(
        len(token_keys),
        dtype=bool,
    )

    for descriptor, vector in zip(
        descriptors,
        descriptor_identities,
    ):

        key = get_token_key(descriptor)
        index = token_to_index[key]

        vector = vector.astype(np.float32)

        if identity_assigned[index]:

            if not np.allclose(
                identity_vectors[index],
                vector,
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError(
                    f"Shared identity inconsistency: {key}"
                )

        else:
            identity_vectors[index] = vector
            identity_assigned[index] = True

    if not identity_assigned.all():
        raise ValueError(
            "One or more token identities are missing."
        )

    if not np.isfinite(identity_vectors).all():
        raise ValueError(
            "Identity vectors contain NaN or Inf."
        )

    print("Identity matrix:", identity_vectors.shape)

    # -----------------------------------------------------
    # G. Fit preprocessing and transform partitions
    # -----------------------------------------------------

    print("[6/10] Preprocessing NSL-KDD...")

    (
        nsl_processor,
        nsl_train_processed,
        nsl_validation_processed,
        nsl_test_processed,
    ) = preprocess_dataset(
        "nsl_kdd",
        nsl_split["X_train"],
        nsl_split["X_validation"],
        nsl_original_test,
    )

    print("[7/10] Preprocessing UNSW-NB15...")

    (
        unsw_processor,
        unsw_train_processed,
        unsw_validation_processed,
        unsw_test_processed,
    ) = preprocess_dataset(
        "unsw_nb15",
        unsw_split["X_train"],
        unsw_split["X_validation"],
        unsw_original_test,
    )

    # -----------------------------------------------------
    # H. Construct token representations
    # -----------------------------------------------------

    print("[8/10] Constructing token representations...")

    nsl_train_tokens = create_token_data(
        nsl_train_processed,
        nsl_split["X_train"].columns,
        token_keys,
        token_to_index,
        nsl_mapping,
    )

    nsl_validation_tokens = create_token_data(
        nsl_validation_processed,
        nsl_split["X_validation"].columns,
        token_keys,
        token_to_index,
        nsl_mapping,
    )

    nsl_test_tokens = create_token_data(
        nsl_test_processed,
        nsl_original_test.columns,
        token_keys,
        token_to_index,
        nsl_mapping,
    )

    unsw_train_tokens = create_token_data(
        unsw_train_processed,
        unsw_split["X_train"].columns,
        token_keys,
        token_to_index,
        unsw_mapping,
    )

    unsw_validation_tokens = create_token_data(
        unsw_validation_processed,
        unsw_split["X_validation"].columns,
        token_keys,
        token_to_index,
        unsw_mapping,
    )

    unsw_test_tokens = create_token_data(
        unsw_test_processed,
        unsw_original_test.columns,
        token_keys,
        token_to_index,
        unsw_mapping,
    )

    # -----------------------------------------------------
    # I. Identify independent original test records
    # -----------------------------------------------------

    print("[9/10] Checking original test overlap...")

    nsl_independent_mask = independent_test_mask(
        nsl_original_train,
        nsl_original_test,
    )

    unsw_independent_mask = independent_test_mask(
        unsw_original_train,
        unsw_original_test,
    )

    print("NSL-KDD full test:", len(nsl_original_test))
    print(
        "NSL-KDD independent test:",
        int(nsl_independent_mask.sum()),
    )
    print(
        "NSL-KDD overlapping test:",
        int((~nsl_independent_mask).sum()),
    )

    print("UNSW-NB15 full test:", len(unsw_original_test))
    print(
        "UNSW-NB15 independent test:",
        int(unsw_independent_mask.sum()),
    )
    print(
        "UNSW-NB15 overlapping test:",
        int((~unsw_independent_mask).sum()),
    )

    # -----------------------------------------------------
    # J. Assemble artifacts
    # -----------------------------------------------------

    nsl_artifact = make_artifact(
        "nsl_kdd",
        token_keys,
        nsl_split["X_train"].columns,
        nsl_train_tokens,
        nsl_validation_tokens,
        nsl_test_tokens,
        nsl_split["y_train"],
        nsl_split["y_validation"],
        nsl_original_y_test,
        nsl_independent_mask,
        nsl_processor,
        identity_dimension,
    )

    unsw_artifact = make_artifact(
        "unsw_nb15",
        token_keys,
        unsw_split["X_train"].columns,
        unsw_train_tokens,
        unsw_validation_tokens,
        unsw_test_tokens,
        unsw_split["y_train"],
        unsw_split["y_validation"],
        unsw_original_y_test,
        unsw_independent_mask,
        unsw_processor,
        identity_dimension,
    )

    # -----------------------------------------------------
    # K. Validate artifacts before saving
    # -----------------------------------------------------

    print("\n[10/10] Validating generated artifacts...")

    validate_artifact(
        nsl_artifact,
        EXPECTED_NSL_FEATURES,
    )

    validate_artifact(
        unsw_artifact,
        EXPECTED_UNSW_FEATURES,
    )

    print("NSL-KDD artifact validation: PASS")
    print("UNSW-NB15 artifact validation: PASS")

    # -----------------------------------------------------
    # L. Save artifacts
    # -----------------------------------------------------

    print("\nSaving artifacts...")

    joblib.dump(
        nsl_artifact,
        unified_artifact_path(config, "nsl_kdd"),
        compress=3,
    )

    joblib.dump(
        unsw_artifact,
        OUTPUT_DIR / "unsw_nb15_unified.joblib",
        compress=3,
    )

    joblib.dump(
        nsl_processor,
        OUTPUT_DIR / "nsl_kdd_preprocessor.joblib",
    )

    joblib.dump(
        unsw_processor,
        OUTPUT_DIR / "unsw_nb15_preprocessor.joblib",
    )

    np.save(
        OUTPUT_DIR / "feature_identity_vectors.npy",
        identity_vectors,
    )

    # Save the complete fitted encoder, including TF-IDF state.
    joblib.dump(
        identity_encoder,
        OUTPUT_DIR / "feature_identity_encoder.joblib",
    )

    # Also save neural-network weights separately.
    torch.save(
        identity_encoder.state_dict(),
        OUTPUT_DIR / "feature_identity_encoder.pt",
    )

    # Save schema.
    schema = {
        "schema_version": 2,
        "source_identity_count": len(token_keys),
        "identity_dimension": identity_dimension,
        "representation": (
            "78 source identity slots"
        ),
        "extension_identity_encoder_saved": True,
        "extension_identity_method": (
            "FeatureIdentityEncoder using feature descriptors"
        ),
        "dynamic_extension_value_tokens_implemented": False,
        "mask_definitions": {
            "M_avail": (
                "1 when feature exists in the dataset; "
                "0 when structurally unavailable."
            ),
            "M_miss": (
                "1 when an existing measurement is missing; "
                "0 when observed."
            ),
        },
        "unavailable_value_policy": (
            "Unavailable slots use storage placeholders only. "
            "They must be replaced with learned missing-feature "
            "representations before neural-network input."
        ),
        "statistics_source": (
            "audited_source_training_partitions_only"
        ),
        "features": schema_features,
    }

    with open(
        OUTPUT_DIR / "unified_feature_schema.yaml",
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            schema,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    # -----------------------------------------------------
    # M. Save report
    # -----------------------------------------------------

    report = [
        "NOVEL-IDS UNIFIED REPRESENTATION REPORT",
        "=" * 65,
        f"Source identity slots: {len(token_keys)}",
        f"Identity dimension: {identity_dimension}",
        "Descriptor statistics: training partitions only",
        "",
    ]

    for name, artifact in [
        ("NSL-KDD", nsl_artifact),
        ("UNSW-NB15", unsw_artifact),
    ]:

        report.append(name)

        for split_name in [
            "train",
            "validation",
            "test_full",
            "test_independent",
        ]:

            split = artifact[split_name]

            available_count = int(
                split["M_avail"][0].sum()
            )

            unavailable_count = (
                len(token_keys) - available_count
            )

            missing_count = int(
                split["M_miss"].sum()
            )

            report.append(
                f"{split_name}: {len(split['labels']):,} records"
            )

            report.append(
                f"  Available features: {available_count}"
            )

            report.append(
                f"  Structurally unavailable features: "
                f"{unavailable_count}"
            )

            report.append(
                f"  Missing measurements: {missing_count}"
            )

        report.append("")

    report.extend([
        "Preprocessing fitted on training partitions only.",
        "Validation and test partitions were transform-only.",
        "Actual available feature measurements were preserved.",
        "Unavailable features were not zero-filled.",
        "M_avail and M_miss were stored separately.",
        "Full and independent original test partitions were retained.",
        "Dynamic unseen value-token construction is not yet implemented.",
        "Learned missing-feature representations are not yet implemented.",
    ])

    report_text = "\n".join(report)

    report_path = (
        RESULTS_DIR / "unified_representation_report.txt"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\n" + report_text)

    print("\n" + "=" * 70)
    print("UNIFIED REPRESENTATION BUILD FINISHED")
    print("=" * 70)

    print("Output directory:", OUTPUT_DIR)
    print("Schema:", OUTPUT_DIR / "unified_feature_schema.yaml")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
