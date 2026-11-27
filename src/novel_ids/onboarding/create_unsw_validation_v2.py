"""Create a versioned UNSW validation subset excluding official-test features.

This module only filters existing validation-aligned artifacts. It never fits
preprocessing or changes source rows, labels, test data, or source scalers.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from novel_ids.onboarding.exact_feature_matching import (
    exact_feature_key,
    exact_feature_overlap_groups,
)
from novel_ids.onboarding.load_source_datasets import (
    PROJECT_ROOT,
    unified_artifact_path,
    validation_split_path,
)


CONFIG_PATH = PROJECT_ROOT / "config/datasets.yaml"
SPLIT_REL = Path("data/interim/validation_splits/unsw_nb15_split_v2_no_test_feature_overlap.joblib")
UNIFIED_REL = Path("data/processed/unified_representation/unsw_nb15_unified_v2_no_test_feature_overlap.joblib")
MANIFEST_REL = Path("data/interim/validation_splits/unsw_nb15_validation_v2_exclusion_manifest.json")
INDEX_DIR_REL = Path("data/processed/evaluation_indices")
ORIGINAL_SPLIT_REL = Path("data/interim/validation_splits/unsw_nb15_split.joblib")
ORIGINAL_UNIFIED_REL = Path("data/processed/unified_representation/unsw_nb15_unified.joblib")
EXCLUDED_COLUMNS = {"id", "attack_cat", "label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _position_map(frame: pd.DataFrame) -> dict[tuple, list[int]]:
    result: dict[tuple, list[int]] = defaultdict(list)
    for pos, row in enumerate(frame.itertuples(index=False, name=None)):
        result[exact_feature_key(row)].append(pos)
    return result


def _duplicates(position_map: dict[tuple, list[int]]) -> dict[str, int]:
    groups = [positions for positions in position_map.values() if len(positions) > 1]
    return {
        "groups": len(groups),
        "excess_rows": sum(len(positions) - 1 for positions in groups),
    }


def _conflict_summary(left, right, left_labels, right_labels, fields):
    conflicts = {field: {"groups": 0, "occurrence_pairs": 0, "right_positions": set()} for field in fields}
    for key in left.keys() & right.keys():
        li, ri = left[key], right[key]
        for field in fields:
            lvals = [str(left_labels[field][i]) for i in li]
            rvals = [str(right_labels[field][i]) for i in ri]
            all_values = set(lvals + rvals)
            if len(all_values) > 1:
                conflicts[field]["groups"] += 1
                conflicts[field]["occurrence_pairs"] += sum(a != b for a in lvals for b in rvals)
                conflicts[field]["right_positions"].update(ri)
    return conflicts


def _audit(left, right, left_labels, right_labels, fields, left_name, right_name):
    lm, rm = _position_map(left), _position_map(right)
    groups = exact_feature_overlap_groups(left, right)
    matching_left = sorted({p for g in groups for p in g["left_positions"]})
    matching_right = sorted({p for g in groups for p in g["right_positions"]})
    conflicts = _conflict_summary(lm, rm, left_labels, right_labels, fields)
    conflicted_right_rows = set().union(*(conflicts[f]["right_positions"] for f in fields)) if fields else set()
    # Output source-index traceability as group records, without including feature payloads.
    matched = []
    for number, group in enumerate(groups, 1):
        matched.append({
            "group_id": number,
            "left_positions": group["left_positions"],
            "right_positions": group["right_positions"],
            "label_values_left": {f: sorted({str(left_labels[f][p]) for p in group["left_positions"]}) for f in fields},
            "label_values_right": {f: sorted({str(right_labels[f][p]) for p in group["right_positions"]}) for f in fields},
        })
    conflict_groups = {}
    for field in fields:
        conflict_groups[field] = [g for g in matched if len(set(g["label_values_left"][field] + g["label_values_right"][field])) > 1]
    return {
        "left": left_name,
        "right": right_name,
        "rows_left": len(left),
        "rows_right": len(right),
        "shared_unique_feature_vectors": len(groups),
        "matching_rows_left": len(matching_left),
        "matching_rows_right": len(matching_right),
        "duplicates_left": _duplicates(lm),
        "duplicates_right": _duplicates(rm),
        "label_fields": fields,
        "label_conflicts": {
            f: {
                "groups": conflicts[f]["groups"],
                "cross_occurrence_pairs": conflicts[f]["occurrence_pairs"],
                "distinct_right_rows": len(conflicts[f]["right_positions"]),
                "right_row_percent": 100 * len(conflicts[f]["right_positions"]) / len(right) if len(right) else 0,
            }
            for f in fields
        },
        "any_label_conflict": {
            "distinct_right_rows": len(conflicted_right_rows),
            "right_row_percent": 100 * len(conflicted_right_rows) / len(right) if len(right) else 0,
        },
        "groups": matched,
        "conflicting_groups": conflict_groups,
    }


def _write_index(path: Path, dataset: str, selection: str, source_path: Path, source_hash: str,
                 feature_columns: list[str], n_rows: int, indices: list[int], excluded: int,
                 groups: int) -> dict:
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "selection": selection,
        "purpose": "evaluation row-selection list; does not modify test data",
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "feature_columns": feature_columns,
        "excluded_feature_columns": sorted(EXCLUDED_COLUMNS) if dataset == "unsw_nb15" else ["attack_label", "difficulty"],
        "matching": "exact typed feature tuple; lossless float.hex; candidate buckets verified by complete key equality",
        "source_row_count": n_rows,
        "selected_row_count": len(indices),
        "excluded_row_count": excluded,
        "matching_feature_group_count": groups,
        "source_indices": indices,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(path),
            "row_count": len(indices), "excluded_row_count": excluded,
            "matching_feature_group_count": groups}


def create_artifacts() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    unsw = config["datasets"]["unsw_nb15"]
    raw_train_path, raw_test_path = Path(unsw["train"]), Path(unsw["test"])
    train, test = pd.read_csv(raw_train_path), pd.read_csv(raw_test_path)
    features = [c for c in train.columns if c not in EXCLUDED_COLUMNS]
    if len(features) != 42 or features != [c for c in test.columns if c not in EXCLUDED_COLUMNS]:
        raise ValueError("Configured UNSW raw feature schema is not the expected ordered 42 features.")
    X_train, X_test = train[features], test[features]

    # Always derive the new version from the protected original artifacts,
    # even after config points to this v2 output. This makes reruns idempotent.
    split_path = PROJECT_ROOT / ORIGINAL_SPLIT_REL
    original = joblib.load(split_path)
    original_validation = original["X_validation"]
    original_validation.columns = features
    val_labels = original["y_validation"].reset_index(drop=True)
    # The split generator preserves source DataFrame indices. Verify the
    # original validation values and labels against those exact source rows.
    source_indices = original_validation.index.to_numpy(dtype=np.int64)
    if len(np.unique(source_indices)) != len(source_indices) or np.any(source_indices < 0) or np.any(source_indices >= len(train)):
        raise ValueError("Original validation source indices are missing, duplicated, or out of range.")
    pd.testing.assert_frame_equal(
        original_validation.reset_index(drop=True),
        train.iloc[source_indices][features].reset_index(drop=True),
        check_dtype=True,
    )
    expected_val_labels = train.iloc[source_indices]["attack_cat"].astype(str).reset_index(drop=True)
    pd.testing.assert_series_equal(val_labels.astype(str), expected_val_labels, check_names=False)
    split_train_indices = original["X_train"].index.to_numpy(dtype=np.int64)
    if len(np.unique(split_train_indices)) != len(split_train_indices) or np.any(split_train_indices < 0) or np.any(split_train_indices >= len(train)):
        raise ValueError("Saved training source indices are missing, duplicated, or out of range.")
    if set(source_indices) & set(split_train_indices):
        raise ValueError("Saved training and validation source indices overlap.")
    if sorted(np.concatenate([source_indices, split_train_indices]).tolist()) != list(range(len(train))):
        raise ValueError("Saved training/validation indices do not cover the official training source exactly.")
    split_train_features = original["X_train"].copy()
    split_train_features.columns = features
    pd.testing.assert_frame_equal(
        split_train_features.reset_index(drop=True),
        train.iloc[split_train_indices][features].reset_index(drop=True),
        check_dtype=True,
    )
    expected_train_labels = train.iloc[split_train_indices]["attack_cat"].astype(str).reset_index(drop=True)
    pd.testing.assert_series_equal(
        original["y_train"].astype(str).reset_index(drop=True),
        expected_train_labels,
        check_names=False,
    )
    overlaps = exact_feature_overlap_groups(original_validation.reset_index(drop=True), X_test.reset_index(drop=True))
    test_by_val = {}
    group_for_val = {}
    for gid, group in enumerate(overlaps, 1):
        for vp in group["left_positions"]:
            test_by_val[vp] = group["right_positions"]
            group_for_val[vp] = gid
    excluded = sorted(test_by_val)
    keep = np.ones(len(original_validation), dtype=bool)
    keep[excluded] = False
    retained_positions = np.flatnonzero(keep).tolist()
    new_split = dict(original)
    new_split["X_validation"] = original["X_validation"].iloc[retained_positions].copy()
    new_split["y_validation"] = original["y_validation"].iloc[retained_positions].copy()
    new_split["validation_original_positions"] = retained_positions
    new_split["validation_source_row_indices"] = [int(original_validation.index[i]) for i in retained_positions]
    new_split["validation_artifact_version"] = "v2_no_test_feature_overlap"
    new_split["source_validation_artifact"] = split_path.relative_to(PROJECT_ROOT).as_posix()
    new_split_path = PROJECT_ROOT / SPLIT_REL
    new_split_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_split, new_split_path, compress=3)

    # Verify exact positional filtering for original validation values/labels.
    pd.testing.assert_frame_equal(new_split["X_validation"], original["X_validation"].iloc[retained_positions])
    pd.testing.assert_series_equal(new_split["y_validation"], original["y_validation"].iloc[retained_positions])

    # Filter only row-aligned validation fields in existing unified representation.
    old_unified_path = PROJECT_ROOT / ORIGINAL_UNIFIED_REL
    old_unified = joblib.load(old_unified_path)
    old_val = old_unified["validation"]
    if len(old_val["labels"]) != len(original_validation):
        raise ValueError("Unified validation row count does not match source split.")
    new_unified = dict(old_unified)
    new_unified["validation"] = {}
    aligned_fields = []
    for key, value in old_val.items():
        if isinstance(value, pd.Series) and len(value) == len(original_validation):
            new_unified["validation"][key] = value.iloc[retained_positions].copy()
            aligned_fields.append(key)
        elif isinstance(value, np.ndarray) and value.ndim and len(value) == len(original_validation):
            new_unified["validation"][key] = value[retained_positions].copy()
            aligned_fields.append(key)
        else:
            new_unified["validation"][key] = value
    new_unified["validation_original_positions"] = retained_positions
    new_unified["validation_artifact_version"] = "v2_no_test_feature_overlap"
    new_unified_path = PROJECT_ROOT / UNIFIED_REL
    new_unified_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_unified, new_unified_path, compress=3)
    # Assert every partition except validation stayed identical; validation is exact positional subset.
    for part in ("train", "test_full", "test_independent"):
        for key in old_unified[part]:
            if isinstance(old_unified[part][key], pd.Series):
                pd.testing.assert_series_equal(new_unified[part][key], old_unified[part][key])
            else:
                np.testing.assert_array_equal(new_unified[part][key], old_unified[part][key])
    for key, value in old_val.items():
        if key in aligned_fields:
            np.testing.assert_equal(np.asarray(new_unified["validation"][key]), np.asarray(value)[retained_positions])

    # Change only the two active UNSW artifact path values, preserving all
    # other config text and formatting.
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    config_lines = config_text.splitlines()
    start = next(i for i, line in enumerate(config_lines) if line.strip() == "unsw_nb15:")
    end = next((i for i in range(start + 1, len(config_lines)) if config_lines[i] and not config_lines[i].startswith(" ")), len(config_lines))
    block = config_lines[start:end]
    desired = {
        "validation_split": SPLIT_REL.as_posix(),
        "unified_artifact": UNIFIED_REL.as_posix(),
    }
    for key, value in desired.items():
        matches = [i for i, line in enumerate(block) if line.startswith(f"    {key}:")]
        replacement = f'    {key}: "{value}"'
        if matches:
            block[matches[0]] = replacement
        else:
            format_index = next(i for i, line in enumerate(block) if line.startswith("    format:"))
            block.insert(format_index + 1, replacement)
    config_lines[start:end] = block
    CONFIG_PATH.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    # Reload and verify path resolver selects the versioned outputs.
    active_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if validation_split_path(active_config, "unsw_nb15") != new_split_path:
        raise AssertionError("Active config did not resolve to new split.")
    if unified_artifact_path(active_config, "unsw_nb15") != new_unified_path:
        raise AssertionError("Active config did not resolve to new unified artifact.")

    # Official train/test audit and evaluation index selections for UNSW.
    unsw_audit = _audit(X_train.reset_index(drop=True), X_test.reset_index(drop=True),
                        {"attack_cat": train["attack_cat"].reset_index(drop=True), "label": train["label"].reset_index(drop=True)},
                        {"attack_cat": test["attack_cat"].reset_index(drop=True), "label": test["label"].reset_index(drop=True)},
                        ["attack_cat", "label"], "UNSW official train", "UNSW official test")
    train_map = _position_map(X_train)
    test_independent = [i for i, row in enumerate(X_test.itertuples(index=False, name=None)) if exact_feature_key(row) not in train_map]
    index_dir = PROJECT_ROOT / INDEX_DIR_REL
    train_hash, test_hash = sha256(raw_train_path), sha256(raw_test_path)
    unsw_indices = {
        "full": _write_index(index_dir / "unsw_nb15_test_full_v1.json", "unsw_nb15", "full_original_test", raw_test_path, test_hash, features, len(test), list(range(len(test))), 0, len(unsw_audit["groups"])),
        "training_independent": _write_index(index_dir / "unsw_nb15_test_training_independent_v1.json", "unsw_nb15", "training_independent_test", raw_test_path, test_hash, features, len(test), test_independent, len(test)-len(test_independent), len(unsw_audit["groups"])),
    }

    # NSL official raw audit and evaluation selections.
    nsl = config["datasets"]["nsl_kdd"]
    nsl_train_path, nsl_test_path = Path(nsl["train"]), Path(nsl["test"])
    meta = pd.read_csv(nsl["metadata"], header=None)
    nsl_features = meta.iloc[:, 0].astype(str).str.strip().tolist()
    nsl_train_raw = pd.read_csv(nsl_train_path, header=None, names=nsl_features + ["attack_label", "difficulty"])
    nsl_test_raw = pd.read_csv(nsl_test_path, header=None, names=nsl_features + ["attack_label", "difficulty"])
    nsl_audit = _audit(nsl_train_raw[nsl_features].reset_index(drop=True), nsl_test_raw[nsl_features].reset_index(drop=True),
                       {"attack_label": nsl_train_raw["attack_label"].reset_index(drop=True)},
                       {"attack_label": nsl_test_raw["attack_label"].reset_index(drop=True)},
                       ["attack_label"], "NSL official train", "NSL official KDDTest+")
    nsl_train_map = _position_map(nsl_train_raw[nsl_features])
    nsl_independent = [i for i, row in enumerate(nsl_test_raw[nsl_features].itertuples(index=False, name=None)) if exact_feature_key(row) not in nsl_train_map]
    nsl_hash = sha256(nsl_test_path)
    nsl_indices = {
        "full": _write_index(index_dir / "nsl_kdd_test_full_v1.json", "nsl_kdd", "full_original_test", nsl_test_path, nsl_hash, nsl_features, len(nsl_test_raw), list(range(len(nsl_test_raw))), 0, len(nsl_audit["groups"])),
        "training_independent": _write_index(index_dir / "nsl_kdd_test_training_independent_v1.json", "nsl_kdd", "training_independent_test", nsl_test_path, nsl_hash, nsl_features, len(nsl_test_raw), nsl_independent, len(nsl_test_raw)-len(nsl_independent), len(nsl_audit["groups"])),
    }

    # Exact raw-feature test overlaps and conflicts are also the refreshed raw-only duplicate audit (A and D).
    # C uses original source training values and newly retained raw validation values.
    kept_raw_val = original_validation.reset_index(drop=True).iloc[retained_positions].copy()
    kept_raw_val.columns = features
    c_audit = _audit(split_train_features.reset_index(drop=True), kept_raw_val.reset_index(drop=True),
                     {"attack_cat": original["y_train"].reset_index(drop=True)},
                     {"attack_cat": val_labels.iloc[retained_positions].reset_index(drop=True), "label": pd.Series([None] * len(retained_positions))},
                     ["attack_cat"], "UNSW saved training split", "UNSW retained validation")
    if c_audit["shared_unique_feature_vectors"] or c_audit["matching_rows_left"] or c_audit["matching_rows_right"]:
        raise AssertionError("New UNSW validation still overlaps source training features.")

    # Manifest: every excluded validation row and every test occurrence in its feature group.
    exclusions = []
    conflict_records = []
    for vp in excluded:
        ti = test_by_val[vp]
        source_train_index = int(original_validation.index[vp])
        vl = str(val_labels.iloc[vp])
        validation_binary = int(train.iloc[source_train_index]["label"])
        test_attack = [str(test.iloc[i]["attack_cat"]) for i in ti]
        test_binary = [int(test.iloc[i]["label"]) for i in ti]
        item = {
            "validation_position": vp,
            "validation_source_training_row_index": source_train_index,
            "test_feature_group_id": group_for_val[vp],
            "official_test_source_row_indices": [int(i) for i in ti],
            "validation_attack_cat": vl,
            "validation_binary_label": validation_binary,
            "test_attack_cat_labels": test_attack,
            "test_binary_labels": test_binary,
            "attack_cat_conflict": any(x != vl for x in test_attack),
            "binary_label_conflict": any(x != validation_binary for x in test_binary),
        }
        exclusions.append(item)
    # Precise conflict groups with source indices and labels from official partitions.
    for field in ("attack_cat", "label"):
        conflict_records.append({"field": field, "groups": unsw_audit["conflicting_groups"][field]})

    # Per-class counts and percent removed for original label spellings.
    old_counts = Counter(str(v) for v in original["y_validation"])
    new_counts = Counter(str(v) for v in new_split["y_validation"])
    class_counts = []
    for cls in sorted(old_counts):
        removed = old_counts[cls] - new_counts.get(cls, 0)
        pct = 100.0 * removed / old_counts[cls] if old_counts[cls] else 0.0
        class_counts.append({"class": cls, "original": old_counts[cls], "retained": new_counts.get(cls, 0), "removed": removed,
                             "percent_removed": pct, "flag_ge_10_percent": pct >= 10.0,
                             "absent_after_filter": new_counts.get(cls, 0) == 0})

    # Earlier validation/test profile and corrected active validation exact audit.
    val_test_audit = _audit(original_validation.reset_index(drop=True), X_test.reset_index(drop=True),
                            {"attack_cat": val_labels, "label": pd.Series([None]*len(val_labels))},
                            {"attack_cat": test["attack_cat"].reset_index(drop=True), "label": test["label"].reset_index(drop=True)},
                            ["attack_cat"], "UNSW original validation", "UNSW official test")

    manifest = {
        "schema_version": 1,
        "dataset": "UNSW-NB15",
        "version": "v2_no_test_feature_overlap",
        "matching_definition": {"feature_columns": features, "excluded_columns": sorted(EXCLUDED_COLUMNS),
                                "method": "typed tuple keys; float values encoded by float.hex; exact key equality is verified within any candidate hash bucket; no rounding or coercion"},
        "original_validation": {"path": split_path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(split_path), "row_count": len(original_validation)},
        "new_validation": {"path": SPLIT_REL.as_posix(), "sha256": sha256(new_split_path), "row_count": len(retained_positions)},
        "original_unified": {"path": old_unified_path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(old_unified_path)},
        "new_unified": {"path": UNIFIED_REL.as_posix(), "sha256": sha256(new_unified_path), "filtered_validation_fields": aligned_fields},
        "official_training_source": {"path": str(raw_train_path), "row_count": len(train), "sha256": train_hash},
        "official_test_source": {"path": str(raw_test_path), "row_count": len(test), "sha256": test_hash},
        "row_counts": {"original_validation": len(original_validation), "retained_validation": len(retained_positions), "excluded_validation": len(excluded), "matching_feature_groups": len(overlaps)},
        "source_split_provenance": {
            "training_source_path": str(raw_train_path),
            "training_source_content_identity": "canonical UNSW-NB15 175,341-row training partition; local filename is UNSW_NB15_testing-set.csv",
            "split_method": "StratifiedGroupKFold candidate selection over training source only",
            "folds": 5,
            "shuffle": True,
            "random_state": int(original.get("random_state", 42)),
            "target_validation_fraction": float(original.get("validation_fraction", 0.2)),
            "grouping": "pandas hash_pandas_object over raw feature rows in original split-generation code; the resulting current saved rows were independently checked by exact source-row feature and attack_cat equality",
            "index_handling": "saved DataFrame indices are source row positions; source indices are unique, in range, disjoint, and together cover all official training rows",
            "source_row_feature_and_label_comparison": "PASS for all saved training and original validation rows",
        },
        "excluded_rows": exclusions,
        "label_conflicting_cross_partition_groups": conflict_records,
        "class_counts": class_counts,
        "audits": {"A_unsw_train_vs_test": unsw_audit, "B_nsl_train_vs_test": nsl_audit,
                   "C_unsw_train_vs_retained_validation": c_audit, "D_unsw_raw_feature_only_profile": unsw_audit,
                   "old_unsw_validation_vs_test_superseded": val_test_audit},
        "evaluation_index_files": {"unsw_nb15": unsw_indices, "nsl_kdd": nsl_indices},
        "preservation": "No labels, training rows, official test rows, raw sources, original validation artifacts, or fitted preprocessing were modified; new artifacts filter existing validation-aligned fields only.",
    }
    manifest_path = PROJECT_ROOT / MANIFEST_REL
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256(manifest_path)
    checksum_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(manifest_path)}  {manifest_path.name}\n", encoding="ascii")
    print(json.dumps({k: manifest[k] for k in ("row_counts", "class_counts", "audits", "evaluation_index_files")}, indent=2))
    print(f"new_split_sha256={sha256(new_split_path)}")
    print(f"new_unified_sha256={sha256(new_unified_path)}")
    print(f"manifest_sha256={sha256(manifest_path)}")
    print(f"manifest_path={MANIFEST_REL.as_posix()}")
    return manifest


if __name__ == "__main__":
    create_artifacts()
