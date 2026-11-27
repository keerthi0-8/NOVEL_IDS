import numpy as np
import pytest

from novel_ids.residual.stage6_probes import fit_logistic_probe_pair


def test_paired_logistic_probes_fit_on_train_and_report_validation_support():
    rng = np.random.default_rng(70)
    y_train = np.repeat(np.array(["normal", "dos", "probe"]), 80)
    y_val = np.repeat(np.array(["normal", "dos", "probe"]), 25)
    centers = {
        "normal": np.array([-2.0, 0.0, 0.0, 0.0]),
        "dos": np.array([2.0, 0.0, 0.0, 0.0]),
        "probe": np.array([0.0, 2.0, 0.0, 0.0]),
    }
    x_train = np.stack([centers[y] for y in y_train]) + rng.normal(0, 0.25, (len(y_train), 4))
    x_val = np.stack([centers[y] for y in y_val]) + rng.normal(0, 0.25, (len(y_val), 4))
    r_train = np.stack([
        np.array([0.0, 0.0, 2.0 if y == "probe" else 0.0, 0.0])
        for y in y_train
    ]) + rng.normal(0, 0.2, (len(y_train), 4))
    r_val = np.stack([
        np.array([0.0, 0.0, 2.0 if y == "probe" else 0.0, 0.0])
        for y in y_val
    ]) + rng.normal(0, 0.2, (len(y_val), 4))

    result = fit_logistic_probe_pair(
        train_z_s=x_train,
        train_r=r_train,
        train_labels=y_train,
        validation_z_s=x_val,
        validation_r=r_val,
        validation_labels=y_val,
        seed=42,
    )
    assert set(result) == {"Z_s", "[Z_s,R]", "macro_f1_gain"}
    assert np.isfinite(result["macro_f1_gain"])
    assert set(result["Z_s"]["validation_class_support"]) == {"normal", "dos", "probe"}
    assert result["[Z_s,R]"]["train_rows"] == len(y_train)
    assert result["[Z_s,R]"]["validation_rows"] == len(y_val)
    assert all(
        item["support"] == 25
        for item in result["[Z_s,R]"]["per_class"].values()
    )


def test_probe_rejects_nonfinite_features_and_single_class_training():
    x_train = np.ones((8, 3), dtype=np.float32)
    x_val = np.ones((4, 3), dtype=np.float32)
    r_train = np.ones((8, 2), dtype=np.float32)
    r_val = np.ones((4, 2), dtype=np.float32)
    y_train = np.array(["normal"] * 8)
    y_val = np.array(["normal"] * 4)

    with pytest.raises(ValueError, match="at least two"):
        fit_logistic_probe_pair(
            train_z_s=x_train, train_r=r_train, train_labels=y_train,
            validation_z_s=x_val, validation_r=r_val, validation_labels=y_val,
        )

    y_train = np.array(["normal", "dos"] * 4)
    x_train[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        fit_logistic_probe_pair(
            train_z_s=x_train, train_r=r_train, train_labels=y_train,
            validation_z_s=x_val, validation_r=r_val, validation_labels=y_val,
        )


import joblib

from novel_ids.residual.stage6_probes import run_source_validation_probes


def test_source_probes_load_labels_from_separate_label_only_artifacts(tmp_path):
    label_paths = {}
    for name in ("nsl_kdd", "unsw_nb15"):
        path = tmp_path / f"{name}_labels.joblib"
        train_labels = np.array(["normal"] * 20 + ["attack"] * 20)
        validation_labels = np.array(["normal"] * 10 + ["attack"] * 10)
        joblib.dump({
            "schema_version": 1,
            "dataset": name,
            "split_names": ["train", "validation"],
            "train": {"labels": train_labels},
            "validation": {"labels": validation_labels},
            "metadata": {
                "allowed_splits": ["train", "validation"],
                "contains_only_labels": True,
            },
        }, path)
        label_paths[name] = str(path)

    class FakeTrainer:
        provenance = {"source_label_artifact_paths": label_paths}

        def extract_features(self, dataset, split):
            rows = 40 if split == "train" else 20
            labels = np.array(["normal"] * (rows // 2) + ["attack"] * (rows // 2))
            x = np.zeros((rows, 3), dtype=np.float32)
            x[:, 0] = np.where(labels == "normal", -1.0, 1.0)
            return x, np.zeros((rows, 2), dtype=np.float32)

    result = run_source_validation_probes(FakeTrainer(), seed=42)
    assert set(result) == {"nsl_kdd", "unsw_nb15", "mean_probe_gain", "probe_seed", "label_policy"}
    assert result["nsl_kdd"]["label_artifact_metadata"]["contains_only_labels"] is True
    assert result["unsw_nb15"]["Z_s"]["validation_rows"] == 20
