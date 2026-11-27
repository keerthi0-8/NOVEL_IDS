from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import yaml

from novel_ids.onboarding.build_unified_representation import (
    make_test_independent,
)
from novel_ids.onboarding.load_source_datasets import (
    unified_artifact_path,
    validation_split_path,
)


ROOT = Path(__file__).resolve().parents[1]


def test_unsw_approved_assignment_matches_existing_train_validation_artifact():
    with (ROOT / "config" / "datasets.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    unsw = config["datasets"]["unsw_nb15"]
    assert unsw["train"].endswith("UNSW_NB15_testing-set.csv")
    assert unsw["test"].endswith("UNSW_NB15_training-set.csv")

    # The saved split contains only the approved training partition and its
    # validation subset. It contains no test records.
    split_path = validation_split_path(config, "unsw_nb15")
    split = joblib.load(split_path)
    assert split["X_train"].shape == (140273, 42)
    original_path = ROOT / "data" / "interim" / "validation_splits" / "unsw_nb15_split.joblib"
    original = joblib.load(original_path)
    positions = np.asarray(split["validation_original_positions"], dtype=int)
    assert split["X_validation"].shape[1] == 42
    assert len(split["X_train"]) == len(original["X_train"]) == 140273
    pd.testing.assert_frame_equal(split["X_train"], original["X_train"])
    pd.testing.assert_series_equal(split["y_train"], original["y_train"])
    pd.testing.assert_frame_equal(split["X_validation"], original["X_validation"].iloc[positions])
    pd.testing.assert_series_equal(split["y_validation"], original["y_validation"].iloc[positions])
    assert len(split["X_train"]) + len(split["X_validation"]) <= 175341
    assert len(split["y_train"]) == 140273
    assert len(split["y_validation"]) == 32999
    assert split["y_train"].nunique() == 10


def test_unsw_saved_independent_mask_and_artifact_subset_are_aligned():
    artifact_path = (
        ROOT
        / "data"
        / "processed"
        / "unified_representation"
        / "unsw_nb15_unified.joblib"
    )
    artifact = joblib.load(artifact_path)
    full = artifact["test_full"]
    independent = artifact["test_independent"]
    mask = np.asarray(full["independent_mask"])

    assert mask.dtype == np.bool_
    assert mask.shape == (82332,)
    assert int(mask.sum()) == 73791
    assert int((~mask).sum()) == 8541

    for key in (
        "numeric_values",
        "categorical_values",
        "M_avail",
        "M_miss",
    ):
        np.testing.assert_array_equal(
            np.asarray(independent[key]),
            np.asarray(full[key])[mask],
        )

    full_labels = np.asarray(full["labels"]).astype(str)
    independent_labels = np.asarray(independent["labels"]).astype(str)
    np.testing.assert_array_equal(independent_labels, full_labels[mask])

    for key in ("numeric_token_mask", "categorical_token_mask"):
        np.testing.assert_array_equal(independent[key], full[key])


def test_active_unsw_v2_validation_is_a_positional_subset_of_original_artifacts():
    with (ROOT / "config" / "datasets.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    active_split_path = validation_split_path(config, "unsw_nb15")
    active_unified_path = unified_artifact_path(config, "unsw_nb15")
    original_split_path = ROOT / "data" / "interim" / "validation_splits" / "unsw_nb15_split.joblib"
    original_unified_path = ROOT / "data" / "processed" / "unified_representation" / "unsw_nb15_unified.joblib"
    active_split, original_split = joblib.load(active_split_path), joblib.load(original_split_path)
    active, original = joblib.load(active_unified_path), joblib.load(original_unified_path)
    positions = np.asarray(active_split["validation_original_positions"], dtype=int)
    assert len(positions) == 32999
    assert np.all(np.diff(positions) > 0)
    pd.testing.assert_frame_equal(active_split["X_validation"], original_split["X_validation"].iloc[positions])
    pd.testing.assert_series_equal(active_split["y_validation"], original_split["y_validation"].iloc[positions])
    assert active_split["validation_artifact_version"] == "v2_no_test_feature_overlap"

    for key, value in original["validation"].items():
        expected = np.asarray(value)[positions] if np.asarray(value).ndim and len(value) == len(original_split["y_validation"]) else np.asarray(value)
        np.testing.assert_array_equal(np.asarray(active["validation"][key]), expected)
    for partition in ("train", "test_full", "test_independent"):
        for key, value in original[partition].items():
            np.testing.assert_array_equal(np.asarray(active[partition][key]), np.asarray(value))

    manifest_path = ROOT / "data" / "interim" / "validation_splits" / "unsw_nb15_validation_v2_exclusion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_counts"] == {
        "original_validation": 35068,
        "retained_validation": 32999,
        "excluded_validation": 2069,
        "matching_feature_groups": 238,
    }
    assert manifest["new_validation"]["sha256"] == _sha256(active_split_path)


def test_versioned_evaluation_index_lists_preserve_source_order_and_checksums():
    manifest_path = ROOT / "data" / "interim" / "validation_splits" / "unsw_nb15_validation_v2_exclusion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "unsw_nb15": (82332, 73791, 8541),
        "nsl_kdd": (22544, 21880, 664),
    }
    for dataset, (full_count, independent_count, excluded_count) in expected.items():
        selections = manifest["evaluation_index_files"][dataset]
        full_meta = selections["full"]
        independent_meta = selections["training_independent"]
        full_path = ROOT / full_meta["path"]
        independent_path = ROOT / independent_meta["path"]
        full = json.loads(full_path.read_text(encoding="utf-8"))
        independent = json.loads(independent_path.read_text(encoding="utf-8"))
        active_unified = joblib.load(unified_artifact_path(yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text(encoding="utf-8")), dataset))
        stored_mask = np.asarray(active_unified["test_full"]["independent_mask"], dtype=bool)
        assert full["source_indices"] == list(range(full_count))
        assert independent["source_indices"] == sorted(independent["source_indices"])
        assert independent["source_indices"] == np.flatnonzero(stored_mask).tolist()
        assert len(independent["source_indices"]) == independent_count
        assert full_count - independent_count == excluded_count
        assert _sha256(full_path) == full_meta["sha256"]
        assert _sha256(independent_path) == independent_meta["sha256"]


def test_unsw_exclusion_manifest_checksum_sidecar_matches_manifest():
    path = ROOT / "data" / "interim" / "validation_splits" / "unsw_nb15_validation_v2_exclusion_manifest.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    recorded_hash = sidecar.read_text(encoding="ascii").split()[0]
    assert _sha256(path) == recorded_hash


def test_make_test_independent_preserves_masked_row_order_and_feature_masks():
    mask = np.array([True, False, True, False, True])
    tokens = {
        "numeric_values": np.arange(10, dtype=np.float32).reshape(5, 2),
        "categorical_values": np.arange(15, dtype=np.int64).reshape(5, 3),
        "M_avail": np.arange(10, dtype=np.float32).reshape(5, 2),
        "M_miss": np.zeros((5, 2), dtype=np.float32),
        "numeric_token_mask": np.array([True, False]),
        "categorical_token_mask": np.array([False, True, True]),
    }
    labels = pd.Series(["a", "b", "c", "d", "e"], index=[10, 11, 12, 13, 14])

    result = make_test_independent(tokens, labels, mask)

    for key in ("numeric_values", "categorical_values", "M_avail", "M_miss"):
        np.testing.assert_array_equal(result[key], tokens[key][mask])
    np.testing.assert_array_equal(
        np.asarray(result["labels"]).astype(str),
        np.asarray(["a", "c", "e"]),
    )
    for key in ("numeric_token_mask", "categorical_token_mask"):
        np.testing.assert_array_equal(result[key], tokens[key])


def test_nsl_training_validation_artifact_preserves_41_inputs():
    with (ROOT / "config" / "datasets.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    split_path = validation_split_path(config, "nsl_kdd")
    split = joblib.load(split_path)
    original_path = ROOT / "data" / "interim" / "validation_splits" / "nsl_kdd_split.joblib"
    original = joblib.load(original_path)
    assert split["X_train"].shape == (100778, 41)
    assert split["X_validation"].shape == (25074, 41)
    assert len(split["X_train"]) + len(split["X_validation"]) == 125852
    assert len(split["y_train"]) == 100778
    assert len(split["y_validation"]) == 25074
    assert split["y_train"].nunique() == 23
    assert original["X_validation"].shape == (25195, 41)
    assert len(split["X_train"]) == len(original["X_train"])
    pd.testing.assert_frame_equal(split["X_train"], original["X_train"])
    pd.testing.assert_series_equal(split["y_train"], original["y_train"])
    positions = np.asarray(split["validation_original_positions"], dtype=int)
    pd.testing.assert_frame_equal(
        split["X_validation"], original["X_validation"].iloc[positions]
    )
    pd.testing.assert_series_equal(
        split["y_validation"], original["y_validation"].iloc[positions]
    )
    manifest_path = ROOT / "data" / "interim" / "validation_splits" / "nsl_kdd_validation_v2_exclusion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = manifest["excluded_rows"]
    assert len(excluded) == 121
    assert sum(item["labels_conflict"] for item in excluded) == 11
    assert all(item["original_KDDTrain_source_row_index"] is not None for item in excluded)
    assert all(item["KDDTest_source_row_indices"] for item in excluded)
    assert manifest["new_validation_artifact"]["row_count"] == 25074
    assert manifest["new_validation_artifact"]["sha256"] == _sha256(split_path)


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_active_nsl_unified_artifact_uses_retained_validation_only():
    with (ROOT / "config" / "datasets.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    split = joblib.load(validation_split_path(config, "nsl_kdd"))
    active_path = unified_artifact_path(config, "nsl_kdd")
    original_path = ROOT / "data" / "processed" / "unified_representation" / "nsl_kdd_unified.joblib"
    active = joblib.load(active_path)
    original = joblib.load(original_path)
    assert len(active["validation"]["labels"]) == 25074
    np.testing.assert_array_equal(
        np.asarray(active["validation"]["labels"]).astype(str),
        np.asarray(split["y_validation"]).astype(str),
    )
    positions = np.asarray(split["validation_original_positions"], dtype=int)
    for key in active["validation"]:
        current = np.asarray(active["validation"][key])
        prior = np.asarray(original["validation"][key])
        expected = prior[positions] if prior.ndim > 0 and len(prior) == 25195 else prior
        np.testing.assert_array_equal(current, expected)
    for partition in ("train", "test_full", "test_independent"):
        assert active[partition].keys() == original[partition].keys()
        for key in active[partition]:
            np.testing.assert_array_equal(
                np.asarray(active[partition][key]),
                np.asarray(original[partition][key]),
            )


def test_saved_identity_artifact_has_the_approved_source_dimensions():
    identity_path = ROOT / "data" / "processed" / "unified_representation" / "feature_identity_vectors.npy"
    vectors = np.load(identity_path, allow_pickle=False)
    assert vectors.shape == (78, 128)
    assert np.isfinite(vectors).all()


def test_nsl_official_partition_config_and_overlap_policy_are_recorded():
    with (ROOT / "config" / "datasets.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    nsl = config["datasets"]["nsl_kdd"]
    assert nsl["train"].endswith("KDDTrain+.txt")
    assert nsl["test"].endswith("KDDTest+.txt")

    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    assert "Matching labels: 610" in progress
    assert "Conflicting labels: 54" in progress
    assert "Full-test and feature-disjoint evaluation partitions are retained." in progress
