from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


INPUT_SPLITS = ("train", "validation")
INPUT_FIELDS = (
    "numeric_values",
    "categorical_values",
    "M_avail",
    "M_miss",
    "numeric_token_mask",
    "categorical_token_mask",
)


def load_stage6_input_artifact(path: str | Path, expected_dataset: str) -> dict[str, Any]:
    """Load a physically split train/validation input artifact; reject any other split."""
    artifact = joblib.load(path)
    required_top = {"schema_version", "dataset", "split_names", "train", "validation", "metadata"}
    if set(artifact) != required_top:
        raise ValueError(
            f"{expected_dataset}: input artifact must contain exactly train/validation "
            "input partitions and metadata; extra or missing top-level fields detected."
        )
    if artifact["schema_version"] != 1 or artifact["dataset"] != expected_dataset:
        raise ValueError(f"{expected_dataset}: input artifact schema or dataset mismatch.")
    if tuple(artifact["split_names"]) != INPUT_SPLITS:
        raise ValueError(f"{expected_dataset}: only train and validation splits are permitted.")
    metadata = artifact["metadata"]
    if metadata.get("allowed_splits") != list(INPUT_SPLITS):
        raise ValueError(f"{expected_dataset}: input artifact split metadata mismatch.")
    if metadata.get("contains_labels") is not False:
        raise ValueError(f"{expected_dataset}: input artifact must not contain labels.")
    result: dict[str, Any] = {}
    for split in INPUT_SPLITS:
        arrays = artifact[split]
        if set(arrays) != set(INPUT_FIELDS):
            raise ValueError(
                f"{expected_dataset}/{split}: unexpected or missing fields in input artifact."
            )
        result[split] = arrays
    return result


def load_stage6_label_artifact(path: str | Path, expected_dataset: str) -> dict[str, Any]:
    """Load the separate train/validation label artifact used only by probes."""
    artifact = joblib.load(path)
    required_top = {"schema_version", "dataset", "split_names", "train", "validation", "metadata"}
    if set(artifact) != required_top:
        raise ValueError(
            f"{expected_dataset}: label artifact must contain exactly train/validation "
            "labels and metadata; extra or missing top-level fields detected."
        )
    if artifact["schema_version"] != 1 or artifact["dataset"] != expected_dataset:
        raise ValueError(f"{expected_dataset}: label artifact schema or dataset mismatch.")
    if tuple(artifact["split_names"]) != INPUT_SPLITS:
        raise ValueError(f"{expected_dataset}: label artifact may contain train/validation only.")
    metadata = artifact["metadata"]
    if metadata.get("allowed_splits") != list(INPUT_SPLITS):
        raise ValueError(f"{expected_dataset}: label artifact split metadata mismatch.")
    if metadata.get("contains_only_labels") is not True:
        raise ValueError(f"{expected_dataset}: label artifact must contain labels only.")
    for split in INPUT_SPLITS:
        if set(artifact[split]) != {"labels"}:
            raise ValueError(
                f"{expected_dataset}/{split}: label artifact must contain labels only."
            )
    return artifact
