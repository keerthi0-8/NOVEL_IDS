from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from novel_ids.onboarding.exact_feature_matching import exact_feature_overlap_groups
from novel_ids.onboarding.load_source_datasets import load_nsl_kdd, load_config, load_unsw_nb15, validation_split_path


def test_exact_feature_overlap_detects_positive_and_accepts_disjoint_negative():
    left = pd.DataFrame({"n": [1.0, 2.0], "c": ["x", "y"]})
    overlapping = pd.DataFrame({"n": [1.0, 1.0, 3.0], "c": ["x", "x", "z"]})
    disjoint = pd.DataFrame({"n": [8.0, 9.0], "c": ["x", "y"]})

    positive = exact_feature_overlap_groups(left, overlapping)
    assert len(positive) == 1
    assert positive[0] == {"left_positions": [0], "right_positions": [0, 1]}
    assert exact_feature_overlap_groups(left, disjoint) == []


def test_exact_feature_match_handles_missing_and_signed_infinities():
    left = pd.DataFrame({"value": [np.nan, np.inf, -np.inf]})
    right = pd.DataFrame({"value": [np.nan, -np.inf, np.inf]})
    groups = exact_feature_overlap_groups(left, right)
    assert groups == [
        {"left_positions": [0], "right_positions": [0]},
        {"left_positions": [1], "right_positions": [2]},
        {"left_positions": [2], "right_positions": [1]},
    ]


def test_exact_feature_match_checks_full_typed_key_inside_hash_candidate_bucket():
    left = pd.DataFrame({
        "number": pd.Series([1.0000000000000000, 1.0, 1], dtype=object),
        "category": ["near", "same", "1"],
    })
    right = pd.DataFrame({
        "number": [1.0000000000000002, 1.0, 1.0],
        "category": ["same", "same", "1"],
    })
    # All keys deliberately collide at the candidate stage. Only the one
    # exactly typed-equal float/string row may match; int and float differ.
    groups = exact_feature_overlap_groups(left, right, candidate_hasher=lambda _: 0)
    assert groups == [{"left_positions": [1], "right_positions": [1]}]


def test_active_unsw_validation_has_no_exact_raw_feature_overlap_with_official_test():
    config = load_config()
    split = joblib.load(validation_split_path(config, "unsw_nb15"))
    X_train, _, X_test, _ = load_unsw_nb15(config)
    validation_features = split["X_validation"].copy()
    assert validation_features.shape[1] == 42
    validation_features.columns = list(X_test.columns)
    assert exact_feature_overlap_groups(validation_features.reset_index(drop=True), X_test.reset_index(drop=True)) == []


def test_active_nsl_validation_has_no_exact_raw_feature_overlap_with_official_test():
    config = load_config()
    split = joblib.load(validation_split_path(config, "nsl_kdd"))
    _, _, X_test, _ = load_nsl_kdd(config)

    feature_names = list(X_test.columns)
    assert len(feature_names) == 41
    validation_features = split["X_validation"].copy()
    assert validation_features.shape[1] == 41
    validation_features.columns = feature_names

    # Equality is direct typed-value tuple equality over the raw source inputs.
    assert exact_feature_overlap_groups(validation_features, X_test) == []
