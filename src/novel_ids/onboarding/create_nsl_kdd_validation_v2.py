"""Create the approved NSL-KDD validation artifact excluding KDDTest+ features."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from novel_ids.onboarding.exact_feature_matching import exact_feature_overlap_groups
from novel_ids.onboarding.load_source_datasets import unified_artifact_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"
ORIGINAL_REL = Path("data/interim/validation_splits/nsl_kdd_split.joblib")
NEW_REL = Path(
    "data/interim/validation_splits/nsl_kdd_split_v2_no_kddtest_feature_overlap.joblib"
)
MANIFEST_REL = Path(
    "data/interim/validation_splits/nsl_kdd_validation_v2_exclusion_manifest.json"
)
ORIGINAL_UNIFIED_REL = Path("data/processed/unified_representation/nsl_kdd_unified.joblib")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def create_artifacts() -> tuple[Path, Path, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["datasets"]["nsl_kdd"]
    original_path = PROJECT_ROOT / ORIGINAL_REL
    new_path = PROJECT_ROOT / NEW_REL
    manifest_path = PROJECT_ROOT / MANIFEST_REL
    unified_old_path = PROJECT_ROOT / ORIGINAL_UNIFIED_REL
    unified_new_path = unified_artifact_path({"datasets": {"nsl_kdd": config}}, "nsl_kdd")

    original_hash = sha256(original_path)
    raw_train_path = Path(config["train"])
    raw_test_path = Path(config["test"])
    metadata_path = Path(config["metadata"])
    metadata = pd.read_csv(metadata_path, header=None)
    feature_names = metadata.iloc[:, 0].astype(str).str.strip().tolist()
    if len(feature_names) != 41:
        raise ValueError(f"Expected 41 metadata features, found {len(feature_names)}.")

    source_train = pd.read_csv(raw_train_path, header=None)
    source_test = pd.read_csv(raw_test_path, header=None)
    if source_train.shape[1] != 43 or source_test.shape[1] != 43:
        raise ValueError("NSL-KDD sources must contain 41 features, label, and difficulty.")

    original = joblib.load(original_path)
    X_validation = original["X_validation"]
    y_validation = original["y_validation"]
    if X_validation.shape[1] != 41 or len(X_validation) != len(y_validation):
        raise ValueError("Original NSL-KDD validation artifact has an unexpected schema.")

    # The saved split has positional feature columns; preserve their source order.
    X_validation = X_validation.copy()
    X_validation.columns = range(41)
    validation_features = X_validation.reset_index(drop=True)
    validation_labels = y_validation.reset_index(drop=True).astype(str).str.strip()
    test_features = source_test.iloc[:, :41].copy()
    test_features.columns = range(41)

    overlaps = exact_feature_overlap_groups(validation_features, test_features)
    test_positions_by_validation_position: dict[int, list[int]] = {}
    group_by_validation_position: dict[int, int] = {}
    for group_number, group in enumerate(overlaps, start=1):
        for val_position in group["left_positions"]:
            test_positions_by_validation_position[val_position] = group["right_positions"]
            group_by_validation_position[val_position] = group_number

    excluded_positions = sorted(test_positions_by_validation_position)
    conflict_positions = []
    exclusions = []

    # Recover and validate original KDDTrain+ indices. The split retained its
    # source DataFrame index; exact feature-plus-label comparison verifies it.
    source_train_features = source_train.iloc[:, :41].copy()
    source_train_features.columns = range(41)
    source_train_labels = source_train.iloc[:, 41].astype(str).str.strip().reset_index(drop=True)
    train_match_groups = exact_feature_overlap_groups(
        source_train_features.assign(__attack_label=source_train_labels),
        validation_features.assign(__attack_label=validation_labels),
    )
    source_positions_by_validation: dict[int, list[int]] = defaultdict(list)
    for group in train_match_groups:
        for validation_position in group["right_positions"]:
            source_positions_by_validation[validation_position].extend(group["left_positions"])

    for validation_position in excluded_positions:
        matching_test_positions = test_positions_by_validation_position[validation_position]
        validation_label = str(validation_labels.iloc[validation_position])
        test_labels = [
            str(source_test.iloc[position, 41]).strip()
            for position in matching_test_positions
        ]
        conflict = any(label != validation_label for label in test_labels)
        if conflict:
            conflict_positions.append(validation_position)

        retained_source_candidates = sorted(set(source_positions_by_validation[validation_position]))
        original_dataframe_index = int(original["X_validation"].index[validation_position])
        if (
            0 <= original_dataframe_index < len(source_train)
            and original_dataframe_index in retained_source_candidates
        ):
            recovered_source_index = original_dataframe_index
            source_ambiguity = False
        elif len(retained_source_candidates) == 1:
            recovered_source_index = retained_source_candidates[0]
            source_ambiguity = False
        else:
            recovered_source_index = None
            source_ambiguity = len(retained_source_candidates) != 1

        exclusions.append({
            "original_validation_artifact_row_position": int(validation_position),
            "original_validation_dataframe_index": original_dataframe_index,
            "original_KDDTrain_source_row_index": recovered_source_index,
            "KDDTrain_source_row_candidates": retained_source_candidates,
            "KDDTest_source_row_indices": [int(x) for x in matching_test_positions],
            "validation_label": validation_label,
            "KDDTest_labels": test_labels,
            "labels_conflict": bool(conflict),
            "test_feature_group_id": int(group_by_validation_position[validation_position]),
            "source_row_mapping_ambiguous": bool(source_ambiguity),
            "feature_match_method": "direct exact typed tuple comparison; no hashing or rounding",
            "feature_columns": feature_names,
        })

    # Expected counts are guardrails against changing inputs or ordering.
    if len(excluded_positions) != 121 or len(conflict_positions) != 11:
        raise ValueError(
            f"Expected 121 exclusions and 11 label conflicts; found "
            f"{len(excluded_positions)} and {len(conflict_positions)}."
        )
    ambiguous_exclusions = [item for item in exclusions if item["source_row_mapping_ambiguous"]]

    keep_mask = [position not in test_positions_by_validation_position for position in range(len(X_validation))]
    kept_original_positions = [position for position, keep in enumerate(keep_mask) if keep]
    new_artifact = dict(original)
    new_artifact["X_validation"] = original["X_validation"].iloc[keep_mask].copy()
    new_artifact["y_validation"] = original["y_validation"].iloc[keep_mask].copy()
    new_artifact["validation_original_positions"] = kept_original_positions
    new_artifact["validation_source_row_indices"] = [
        int(original["X_validation"].index[position]) for position in kept_original_positions
    ]
    new_artifact["validation_artifact_version"] = "v2_no_kddtest_feature_overlap"
    new_artifact["source_validation_artifact"] = ORIGINAL_REL.as_posix()

    new_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_artifact, new_path, compress=3)
    new_hash = sha256(new_path)

    # Promote the filtered rows to a versioned unified artifact without
    # refitting preprocessing or altering the existing representation file.
    original_unified_hash = sha256(unified_old_path)
    original_unified = joblib.load(unified_old_path)
    original_unified_validation_count = len(original_unified["validation"]["labels"])
    if original_unified_validation_count != len(X_validation):
        raise ValueError("Unified artifact validation rows do not match the original split.")
    keep_positions = np.asarray(kept_original_positions, dtype=np.int64)
    new_unified = dict(original_unified)
    new_unified["validation"] = {}
    filtered_unified_fields = []
    for name, value in original_unified["validation"].items():
        if isinstance(value, pd.Series) and len(value) == original_unified_validation_count:
            new_unified["validation"][name] = value.iloc[keep_positions].copy()
            filtered_unified_fields.append(name)
        elif isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == original_unified_validation_count:
            new_unified["validation"][name] = value[keep_positions].copy()
            filtered_unified_fields.append(name)
        else:
            new_unified["validation"][name] = value
    if len(new_unified["validation"]["labels"]) != len(new_artifact["y_validation"]):
        raise ValueError("Filtered unified labels have an unexpected row count.")
    if not np.array_equal(
        np.asarray(new_unified["validation"]["labels"]).astype(str),
        np.asarray(new_artifact["y_validation"]).astype(str),
    ):
        raise ValueError("Filtered unified labels do not match the new validation split.")
    new_unified["validation_original_positions"] = kept_original_positions
    new_unified["validation_artifact_version"] = "v2_no_kddtest_feature_overlap"
    unified_new_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_unified, unified_new_path, compress=3)
    unified_new_hash = sha256(unified_new_path)

    original_counts = _counts(y_validation)
    new_counts = _counts(new_artifact["y_validation"])
    removed_counts = {
        label: original_counts.get(label, 0) - new_counts.get(label, 0)
        for label in sorted(set(original_counts) | set(new_counts))
        if original_counts.get(label, 0) != new_counts.get(label, 0)
    }
    matching_test_positions = sorted({
        position
        for group in overlaps
        for position in group["right_positions"]
    })
    source_hashes = {
        "KDDTrain+.txt": {"path": str(raw_train_path), "sha256": sha256(raw_train_path)},
        "KDDTest+.txt": {"path": str(raw_test_path), "sha256": sha256(raw_test_path)},
        "Field Names.csv": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
    }
    manifest = {
        "schema_version": 1,
        "purpose": "exclude validation rows whose exact raw 41-feature vector occurs in official KDDTest+",
        "feature_match_definition": {
            "columns": feature_names,
            "source_columns": "raw positions 0 through 40 in KDDTrain+.txt and KDDTest+.txt",
            "excluded_columns": ["attack_label", "difficulty"],
            "comparison": "direct exact typed tuple equality; no transform, normalization, encoding, rounding, or hash-only decisions",
            "duplicate_handling": "all matching row positions are retained per exact feature group; no occurrence is collapsed",
        },
        "original_validation_artifact": {
            "path": ORIGINAL_REL.as_posix(),
            "sha256": original_hash,
            "row_count": int(len(y_validation)),
        },
        "new_validation_artifact": {
            "path": NEW_REL.as_posix(),
            "sha256": new_hash,
            "row_count": int(len(new_artifact["y_validation"])),
        },
        "original_unified_artifact": {
            "path": ORIGINAL_UNIFIED_REL.as_posix(),
            "sha256": original_unified_hash,
            "validation_row_count": int(original_unified_validation_count),
        },
        "new_unified_artifact": {
            "path": unified_new_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": unified_new_hash,
            "validation_row_count": int(len(new_unified["validation"]["labels"])),
            "filtered_row_aligned_fields": filtered_unified_fields,
        },
        "raw_source_files": source_hashes,
        "excluded_validation_row_count": len(excluded_positions),
        "retained_validation_row_count": len(kept_original_positions),
        "matching_test_feature_group_count": len(overlaps),
        "matching_test_row_count": len(matching_test_positions),
        "label_conflict_count": len(conflict_positions),
        "source_row_mapping_ambiguous_exclusion_count": len(ambiguous_exclusions),
        "original_validation_class_counts": original_counts,
        "new_validation_class_counts": new_counts,
        "removed_rows_by_class": removed_counts,
        "excluded_rows": exclusions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return new_path, manifest_path, manifest


if __name__ == "__main__":
    new_path, manifest_path, manifest = create_artifacts()
    print(f"New validation artifact: {new_path}")
    print(f"New validation SHA-256: {manifest['new_validation_artifact']['sha256']}")
    print(f"Active unified artifact: {manifest['new_unified_artifact']['path']}")
    print(f"Active unified artifact SHA-256: {manifest['new_unified_artifact']['sha256']}")
    print(f"Manifest: {manifest_path}")
    print(f"Validation rows: {manifest['original_validation_artifact']['row_count']} -> {manifest['new_validation_artifact']['row_count']}")
    print(f"Excluded rows: {manifest['excluded_validation_row_count']}; matching test groups: {manifest['matching_test_feature_group_count']}; matching test rows: {manifest['matching_test_row_count']}; label conflicts: {manifest['label_conflict_count']}")
    print(f"Original validation class counts: {manifest['original_validation_class_counts']}")
    print(f"New validation class counts: {manifest['new_validation_class_counts']}")
    print(f"Removed rows by class: {manifest['removed_rows_by_class']}")
