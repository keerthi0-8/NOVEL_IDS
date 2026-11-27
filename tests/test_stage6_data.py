import joblib
import numpy as np
import pytest

from novel_ids.residual.stage6_data import (
    load_stage6_input_artifact,
    load_stage6_label_artifact,
)


def _input_artifact():
    arrays = {
        "numeric_values": np.zeros((3, 78), dtype=np.float32),
        "categorical_values": np.zeros((3, 78), dtype=np.int64),
        "M_avail": np.ones((3, 78), dtype=np.float32),
        "M_miss": np.zeros((3, 78), dtype=np.float32),
        "numeric_token_mask": np.ones(78, dtype=bool),
        "categorical_token_mask": np.zeros(78, dtype=bool),
    }
    return {
        "schema_version": 1,
        "dataset": "nsl_kdd",
        "split_names": ["train", "validation"],
        "train": arrays,
        "validation": arrays,
        "metadata": {
            "allowed_splits": ["train", "validation"],
            "contains_labels": False,
        },
    }


def _label_artifact():
    return {
        "schema_version": 1,
        "dataset": "nsl_kdd",
        "split_names": ["train", "validation"],
        "train": {"labels": np.array(["normal", "dos"])},
        "validation": {"labels": np.array(["normal"])},
        "metadata": {
            "allowed_splits": ["train", "validation"],
            "contains_only_labels": True,
        },
    }


def test_input_artifact_accepts_only_train_validation_features(tmp_path):
    path = tmp_path / "inputs.joblib"
    joblib.dump(_input_artifact(), path)
    loaded = load_stage6_input_artifact(path, "nsl_kdd")
    assert set(loaded) == {"train", "validation"}
    assert set(loaded["train"]) == {
        "numeric_values", "categorical_values", "M_avail", "M_miss",
        "numeric_token_mask", "categorical_token_mask",
    }


def test_input_artifact_rejects_test_partition(tmp_path):
    artifact = _input_artifact()
    artifact["test_full"] = {"numeric_values": np.zeros((1, 78))}
    path = tmp_path / "inputs_with_test.joblib"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="extra or missing top-level fields"):
        load_stage6_input_artifact(path, "nsl_kdd")


def test_input_artifact_rejects_label_fields(tmp_path):
    artifact = _input_artifact()
    artifact["train"]["labels"] = np.array(["normal"])
    path = tmp_path / "inputs_with_labels.joblib"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        load_stage6_input_artifact(path, "nsl_kdd")


def test_label_artifact_accepts_train_validation_labels_only(tmp_path):
    path = tmp_path / "labels.joblib"
    joblib.dump(_label_artifact(), path)
    loaded = load_stage6_label_artifact(path, "nsl_kdd")
    assert set(loaded["train"]) == {"labels"}
    assert set(loaded["validation"]) == {"labels"}


def test_label_artifact_rejects_test_partition(tmp_path):
    artifact = _label_artifact()
    artifact["test_full"] = {"labels": np.array(["attack"])}
    path = tmp_path / "labels_with_test.joblib"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="extra or missing top-level fields"):
        load_stage6_label_artifact(path, "nsl_kdd")


def test_label_artifact_rejects_feature_fields(tmp_path):
    artifact = _label_artifact()
    artifact["train"]["numeric_values"] = np.zeros((2, 78))
    path = tmp_path / "labels_with_features.joblib"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="labels only"):
        load_stage6_label_artifact(path, "nsl_kdd")
