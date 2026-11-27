from pathlib import Path

import numpy as np
import pandas as pd

from novel_ids.inference.api import (
    PROJECT_ROOT,
    _build_source_model_inputs,
    _build_source_slot_maps,
    _load_source_preprocessors,
    _recognize_source_schema,
    _registry,
    onboard_csv,
)


def test_nsl_and_unsw_raw_demo_schemas_are_recognized_with_source_preprocessing():
    expected = {
        "nsl_kdd_validation_sample_unlabeled.csv": "nsl_kdd",
        "unsw_nb15_validation_sample_unlabeled.csv": "unsw_nb15",
    }
    for filename, dataset in expected.items():
        report = onboard_csv(PROJECT_ROOT / "demo_samples" / filename, calibration_rows=500)
        assert report.source_dataset == dataset
        assert report.compatibility["recognized_source_schema"] is True
        assert report.compatibility["compatibility_score"] == 1.0
        assert report.compatibility["trust_level"] == "High"
        assert report.normalization_mode == "source_fitted"
        assert report.model_inputs["numeric_values"].shape == (500, 78)
        assert report.model_inputs["M_avail"].shape == (500, 78)
        assert report.model_inputs["source_preprocessing"]["normalization_fit_on_upload"] is False
        assert report.calibration_frame is None


def test_source_schema_fingerprint_is_case_and_space_insensitive_with_metadata_aliases():
    preprocessors = _load_source_preprocessors()
    frame = pd.read_csv(
        PROJECT_ROOT / "demo_samples" / "nsl_kdd_validation_sample_unlabeled.csv",
        nrows=0,
    )
    columns = []
    for name in frame.columns:
        if name == "protocol_type":
            columns.append("PROTOCOL TYPE")
        elif name == "src_bytes":
            columns.append("Src Bytes")
        else:
            columns.append(name.upper())
    dataset, mapping = _recognize_source_schema(columns, preprocessors)
    assert dataset == "nsl_kdd"
    assert len(mapping) == 41


def test_unknown_schema_keeps_the_existing_unseen_normalization_path(tmp_path):
    path = tmp_path / "unknown.csv"
    pd.DataFrame({
        "unseen_alpha": [1.0, 2.0, 3.0],
        "unseen_beta": ["x", "y", "z"],
    }).to_csv(path, index=False)
    report = onboard_csv(path, calibration_rows=3)
    assert report.source_dataset is None
    assert report.normalization_mode == "option_a_target"
    assert np.all(report.model_inputs["source_feature_indices"] == -1)
    assert np.all(report.model_inputs["M_avail"][:, :78] == 0)


def test_labeled_and_unlabeled_source_files_build_identical_inputs():
    pairs = [
        ("nsl_kdd_validation_sample_unlabeled.csv", "nsl_kdd_validation_sample_labeled.csv", "nsl_kdd"),
        ("unsw_nb15_validation_sample_unlabeled.csv", "unsw_nb15_validation_sample_labeled.csv", "unsw_nb15"),
    ]
    for unlabeled_name, labeled_name, dataset in pairs:
        unlabeled = onboard_csv(PROJECT_ROOT / "demo_samples" / unlabeled_name, calibration_rows=500)
        labeled = onboard_csv(
            PROJECT_ROOT / "demo_samples" / labeled_name,
            label_column="label",
            calibration_rows=500,
        )
        assert unlabeled.source_dataset == labeled.source_dataset == dataset
        for key in ("numeric_values", "categorical_values", "M_avail", "M_miss"):
            assert np.array_equal(unlabeled.model_inputs[key], labeled.model_inputs[key])


def test_source_slot_maps_cover_each_training_preprocessor_feature():
    encoder, descriptors, matching = _registry()
    preprocessors = _load_source_preprocessors()
    maps = _build_source_slot_maps(matching, descriptors, preprocessors)
    for dataset, processor in preprocessors.items():
        assert set(maps[dataset]) == set(processor.feature_names)
        assert len(set(maps[dataset].values())) == len(processor.feature_names)
        assert all(0 <= index < 78 for index in maps[dataset].values())
