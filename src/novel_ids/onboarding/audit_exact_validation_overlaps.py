"""Read-only exact raw-feature overlap audits for active validation splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
import joblib

from novel_ids.onboarding.exact_feature_matching import exact_feature_key, exact_feature_overlap_groups
from novel_ids.onboarding.load_source_datasets import validation_split_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


def _key_counts(frame: pd.DataFrame) -> Counter:
    return Counter(exact_feature_key(row) for row in frame.itertuples(index=False, name=None))


def _audit(name: str, left: pd.DataFrame, right: pd.DataFrame, left_labels, right_labels) -> None:
    groups = exact_feature_overlap_groups(left, right)
    left_counts = _key_counts(left)
    right_counts = _key_counts(right)
    shared_left_rows = sum(len(group["left_positions"]) for group in groups)
    shared_right_rows = sum(len(group["right_positions"]) for group in groups)
    one_to_one_pairs = sum(
        min(len(group["left_positions"]), len(group["right_positions"])) for group in groups
    )
    cartesian_pairs = sum(
        len(group["left_positions"]) * len(group["right_positions"]) for group in groups
    )
    conflict_groups = conflict_pairs = 0
    for group in groups:
        labels_left = [str(left_labels.iloc[pos]) for pos in group["left_positions"]]
        labels_right = [str(right_labels.iloc[pos]) for pos in group["right_positions"]]
        if set(labels_left + labels_right) - {labels_left[0]}:
            conflict_groups += 1
        conflict_pairs += sum(a != b for a in labels_left for b in labels_right)

    print(f"\n{name}")
    print(f"rows_left={len(left)} rows_right={len(right)}")
    print(f"unique_shared_feature_vectors={len(groups)}")
    print(f"matching_rows_left={shared_left_rows} matching_rows_right={shared_right_rows}")
    print(f"one_to_one_occurrence_pairs={one_to_one_pairs} cartesian_occurrence_pairs={cartesian_pairs}")
    print(
        "duplicate_groups_left={} duplicate_excess_rows_left={} duplicate_groups_right={} duplicate_excess_rows_right={}".format(
            sum(count > 1 for count in left_counts.values()),
            sum(count - 1 for count in left_counts.values() if count > 1),
            sum(count > 1 for count in right_counts.values()),
            sum(count - 1 for count in right_counts.values() if count > 1),
        )
    )
    print(f"label_conflict_groups={conflict_groups} conflicting_cross_pairs={conflict_pairs}")
    print(f"zero_exact_feature_overlap={not groups}")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["datasets"]
    nsl_cfg = config["nsl_kdd"]
    unsw_cfg = config["unsw_nb15"]
    nsl_split = joblib.load(validation_split_path(config, "nsl_kdd"))
    unsw_split = joblib.load(validation_split_path(config, "unsw_nb15"))

    nsl_test_raw = pd.read_csv(nsl_cfg["test"], header=None)
    nsl_test_x = nsl_test_raw.iloc[:, :41].copy()
    nsl_test_x.columns = range(41)
    nsl_train_x = nsl_split["X_train"].copy()
    nsl_train_x.columns = range(41)
    nsl_val_x = nsl_split["X_validation"].copy()
    nsl_val_x.columns = range(41)
    nsl_train_labels = nsl_split["y_train"].astype(str).str.strip().reset_index(drop=True)
    nsl_test_labels = nsl_test_raw.iloc[:, 41].astype(str).str.strip().reset_index(drop=True)
    nsl_val_labels = nsl_split["y_validation"].astype(str).str.strip().reset_index(drop=True)

    unsw_train_raw = pd.read_csv(unsw_cfg["train"])
    unsw_test_raw = pd.read_csv(unsw_cfg["test"])
    excluded = {"id", "attack_cat", "label"}
    unsw_features = [column for column in unsw_train_raw.columns if column not in excluded]
    if len(unsw_features) != 42:
        raise ValueError(f"Expected 42 UNSW-NB15 model inputs, found {len(unsw_features)}.")
    if unsw_features != [column for column in unsw_test_raw.columns if column not in excluded]:
        raise ValueError("UNSW-NB15 train/test feature names or order differ.")
    unsw_train_x = unsw_split["X_train"].loc[:, unsw_features].reset_index(drop=True)
    unsw_val_x = unsw_split["X_validation"].loc[:, unsw_features].reset_index(drop=True)
    unsw_test_x = unsw_test_raw.loc[:, unsw_features].reset_index(drop=True)
    unsw_train_labels = unsw_split["y_train"].astype(str).reset_index(drop=True)
    unsw_val_labels = unsw_split["y_validation"].astype(str).reset_index(drop=True)
    unsw_test_labels = unsw_test_raw["attack_cat"].astype(str).reset_index(drop=True)

    _audit("NSL-KDD training vs retained validation (41 raw inputs)", nsl_train_x, nsl_val_x, nsl_train_labels, nsl_val_labels)
    _audit("UNSW-NB15 training vs validation (42 raw inputs)", unsw_train_x, unsw_val_x, unsw_train_labels, unsw_val_labels)
    _audit("UNSW-NB15 validation vs official test (42 raw inputs)", unsw_val_x, unsw_test_x, unsw_val_labels, unsw_test_labels)


if __name__ == "__main__":
    main()
