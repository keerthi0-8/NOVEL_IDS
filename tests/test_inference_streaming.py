import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from novel_ids.inference.api import _predict_input_batch, onboard_csv


def test_streaming_onboarding_uses_bounded_deterministic_unlabeled_sample(tmp_path):
    n = 137
    frame = pd.DataFrame({
        "duration": np.arange(n, dtype=float),
        "bytes": np.arange(n, dtype=float) * 11,
        "protocol": ["tcp", "udp", "icmp"] * (n // 3) + ["tcp", "udp"],
        "all_nan": np.full(n, np.nan),
        "with_inf": np.where(np.arange(n) % 2 == 0, np.inf, np.arange(n, dtype=float)),
        "Label": ["Normal", "DoS"] * (n // 2) + ["Normal"],
    })
    path = tmp_path / "stream.csv"
    frame.to_csv(path, index=False)

    first = onboard_csv(path, label_column="Label", chunksize=13, calibration_rows=17)
    second = onboard_csv(path, label_column="Label", chunksize=13, calibration_rows=17)

    assert first.streaming is True
    assert first.rows == n
    assert first.profile["streaming"] is True
    assert first.profile["chunk_rows"] == 13
    assert first.profile["unlabeled_calibration_sample_rows"] == 17
    assert first.feature_names == ["duration", "bytes", "protocol", "all_nan", "with_inf"]
    assert "Label" not in first.feature_names
    assert first.model_inputs["numeric_values"].shape[0] == 17
    assert first.calibration_frame is not None
    assert len(first.calibration_frame) == 17
    assert first.profile["all_missing_columns"] == ["all_nan"]
    assert first.profile["infinite_numeric_values"] == 69
    pd.testing.assert_frame_equal(first.calibration_frame, second.calibration_frame)
    assert np.isfinite(first.model_inputs["numeric_values"]).all()


def test_streaming_onboarding_small_csv_uses_all_rows_and_excludes_labels(tmp_path):
    frame = pd.DataFrame({
        "x": [1.0, np.inf, np.nan, 4.0],
        "proto": ["tcp", "udp", "unknown", "tcp"],
        "Label": ["Normal", "DoS", "Normal", "DoS"],
    })
    path = tmp_path / "small.csv"
    frame.to_csv(path, index=False)
    report = onboard_csv(path, label_column="Label", chunksize=2, calibration_rows=50)
    assert report.rows == 4
    assert report.model_inputs["numeric_values"].shape[0] == 4
    assert report.profile["input_columns"] == ["x", "proto", "Label"]
    assert report.profile["numeric_feature_names"] == ["x"]
    assert "Label" not in report.feature_names


def test_streaming_onboarding_rejects_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        onboard_csv(path)


def test_new_schema_renamed_shuffled_extra_and_label_invariance(tmp_path):
    n = 19
    features = pd.DataFrame({
        "renamed_duration": np.arange(n, dtype=float),
        "new_protocol": ["tcp", "udp", "unknown_proto"] * 6 + ["unknown_proto"],
        "extra_counter": np.arange(n, dtype=float) * 3,
        "all_nan_extra": np.full(n, np.nan),
        "infinite_extra": np.where(np.arange(n) % 2 == 0, np.inf, np.arange(n, dtype=float)),
    })
    # Deliberately shuffled order and renamed/unseen columns.
    features = features[["new_protocol", "extra_counter", "infinite_extra", "renamed_duration", "all_nan_extra"]]
    unlabeled_path = tmp_path / "renamed_unlabeled.csv"
    labeled_path = tmp_path / "renamed_labeled.csv"
    features.to_csv(unlabeled_path, index=False)
    labeled = features.copy()
    labeled["target_outcome"] = ["Normal", "DoS", "synthetic-unknown"] * 6 + ["Normal"]
    labeled.to_csv(labeled_path, index=False)

    unlabeled = onboard_csv(unlabeled_path, chunksize=5, calibration_rows=19)
    labeled_report = onboard_csv(labeled_path, label_column="target_outcome", chunksize=5, calibration_rows=19)

    assert unlabeled.feature_names == list(features.columns)
    assert labeled_report.feature_names == list(features.columns)
    assert labeled_report.config["feature_columns_in_order"] == list(features.columns)
    assert labeled_report.config["normalization"]["fit_scope"] == "unlabeled target feature rows only"
    assert unlabeled.profile["all_missing_columns"] == ["all_nan_extra"]
    assert unlabeled.profile["infinite_numeric_values"] == 10
    assert unlabeled.model_inputs["numeric_values"].shape[0] == n
    assert np.isfinite(unlabeled.model_inputs["numeric_values"]).all()
    for key in ("numeric_values", "categorical_values", "M_avail", "M_miss"):
        assert np.array_equal(unlabeled.model_inputs[key], labeled_report.model_inputs[key])


def test_onboarding_rejects_missing_requested_label_column(tmp_path):
    path = tmp_path / "missing_label.csv"
    pd.DataFrame({"duration": [1.0, 2.0], "protocol": ["tcp", "udp"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Label column.*not found"):
        onboard_csv(path, label_column="Label")


def test_onboarding_handles_mixed_non_numeric_values_and_malformed_csv(tmp_path):
    mixed_path = tmp_path / "mixed_values.csv"
    pd.DataFrame({"duration_like": ["1.0", "not-a-number", "3.5"], "protocol": ["tcp", "udp", "unknown"]}).to_csv(mixed_path, index=False)
    report = onboard_csv(mixed_path, chunksize=2, calibration_rows=3)
    assert report.rows == 3
    assert "duration_like" in report.feature_names
    assert np.isfinite(report.model_inputs["numeric_values"]).all()

    malformed_path = tmp_path / "malformed.csv"
    malformed_path.write_text('a,b\n1,2\n3,"unterminated\n', encoding="utf-8")
    with pytest.raises(pd.errors.ParserError):
        onboard_csv(malformed_path, chunksize=2)


def test_large_csv_onboarding_keeps_only_bounded_calibration_sample(tmp_path):
    n = 100_000
    path = tmp_path / "large_stream.csv"
    pd.DataFrame({
        "duration": np.arange(n, dtype=np.float64),
        "bytes": np.arange(n, dtype=np.float64) * 7,
        "protocol": np.resize(np.asarray(["tcp", "udp", "icmp", "unknown"]), n),
    }).to_csv(path, index=False)
    report = onboard_csv(path, chunksize=4096, calibration_rows=512)
    assert report.rows == n
    assert report.streaming is True
    assert report.profile["chunk_rows"] == 4096
    assert report.profile["unlabeled_calibration_sample_rows"] == 512
    assert len(report.calibration_frame) == 512
    assert report.frame.empty
    assert report.model_inputs["numeric_values"].shape[0] == 512


def test_prediction_enforces_hard_free_ram_limit_before_model_execution():
    model = SimpleNamespace(
        device=torch.device("cpu"),
        resource_module=SimpleNamespace(free_ram_gb=lambda: 1.5),
        calibration={"temperature": 1.0},
    )
    inputs = {
        "feature_descriptors": [],
        "source_feature_indices": np.asarray([], dtype=np.int64),
        "numeric_values": np.zeros((1, 0), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="HARD STOP: free RAM below 2 GiB"):
        _predict_input_batch(model, inputs, batch_size=1, row_offset=0)
