from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from novel_ids.onboarding.build_unified_representation import (
    build_dataset_feature_mapping,
    build_token_schema,
    create_token_data,
)
from novel_ids.onboarding.load_source_datasets import load_config, validation_split_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_DIR = PROJECT_ROOT / "data" / "processed" / "unified_representation"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "stage6_trainval_only"
TEMP_DIR = PROJECT_ROOT / "data" / "interim" / "stage6_trainval_only_build_tmp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists() or TEMP_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage6 train/validation artifacts: {OUTPUT_DIR} or {TEMP_DIR}"
        )
    TEMP_DIR.mkdir(parents=True, exist_ok=False)

    config = load_config()
    descriptor_path = UNIFIED_DIR / "training_feature_descriptors.yaml"
    descriptors = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))["features"]
    token_keys, token_to_index, _ = build_token_schema(descriptors)

    manifest = {
        "schema_version": 1,
        "allowed_splits": ["train", "validation"],
        "test_partitions_loaded": False,
        "datasets": {},
    }

    try:
        for name in ("nsl_kdd", "unsw_nb15"):
            split_path = validation_split_path(config, name)
            split_hash = sha256_file(split_path)
            split_data = joblib.load(split_path)

            if not {"X_train", "y_train", "X_validation", "y_validation"}.issubset(split_data):
                raise ValueError(f"{name}: validation split artifact is missing required train/validation fields.")

            preprocessor_path = UNIFIED_DIR / f"{name}_preprocessor.joblib"
            preprocessor_hash = sha256_file(preprocessor_path)
            preprocessor = joblib.load(preprocessor_path)

            for key in ("X_train", "X_validation"):
                frame = split_data[key]
                if len(frame.columns) != len(preprocessor.feature_names):
                    raise ValueError(f"{name}/{key}: feature count differs from fitted preprocessor.")
                if name == "nsl_kdd":
                    # The audited NSL-KDD split stores positional columns; restore
                    # the source feature names from the fitted training preprocessor.
                    frame.columns = list(preprocessor.feature_names)
                if list(frame.columns) != list(preprocessor.feature_names):
                    raise ValueError(f"{name}/{key}: feature order differs from fitted preprocessor.")

            mapping = build_dataset_feature_mapping(descriptors, name)
            input_artifact = {
                "schema_version": 1,
                "dataset": name,
                "split_names": ["train", "validation"],
                "metadata": {
                    "allowed_splits": ["train", "validation"],
                    "contains_labels": False,
                    "source_validation_split_path": str(split_path.resolve()),
                    "source_validation_split_sha256": split_hash,
                    "fitted_preprocessor_path": str(preprocessor_path.resolve()),
                    "fitted_preprocessor_sha256": preprocessor_hash,
                    "descriptor_path": str(descriptor_path.resolve()),
                    "descriptor_sha256": sha256_file(descriptor_path),
                    "token_count": len(token_keys),
                },
            }
            label_artifact = {
                "schema_version": 1,
                "dataset": name,
                "split_names": ["train", "validation"],
                "metadata": {
                    "allowed_splits": ["train", "validation"],
                    "contains_only_labels": True,
                    "source_validation_split_path": str(split_path.resolve()),
                    "source_validation_split_sha256": split_hash,
                },
            }

            for split_name, x_key, y_key in (
                ("train", "X_train", "y_train"),
                ("validation", "X_validation", "y_validation"),
            ):
                processed = preprocessor.transform(split_data[x_key])
                tokens = create_token_data(
                    processed,
                    split_data[x_key].columns,
                    token_keys,
                    token_to_index,
                    mapping,
                )
                input_artifact[split_name] = {
                    key: np.asarray(tokens[key])
                    for key in (
                        "numeric_values",
                        "categorical_values",
                        "M_avail",
                        "M_miss",
                        "numeric_token_mask",
                        "categorical_token_mask",
                    )
                }
                labels = np.asarray(split_data[y_key]).astype(str)
                if labels.shape[0] != input_artifact[split_name]["numeric_values"].shape[0]:
                    raise ValueError(f"{name}/{split_name}: labels and inputs row counts differ.")
                label_artifact[split_name] = {"labels": labels}

            input_path = TEMP_DIR / f"{name}_inputs.joblib"
            label_path = TEMP_DIR / f"{name}_labels.joblib"
            joblib.dump(input_artifact, input_path, compress=3)
            joblib.dump(label_artifact, label_path, compress=3)
            manifest["datasets"][name] = {
                "input_artifact": str((OUTPUT_DIR / input_path.name).resolve()),
                "input_sha256": sha256_file(input_path),
                "label_artifact": str((OUTPUT_DIR / label_path.name).resolve()),
                "label_sha256": sha256_file(label_path),
                "train_rows": int(len(label_artifact["train"]["labels"])),
                "validation_rows": int(len(label_artifact["validation"]["labels"])),
                "input_fields": sorted(input_artifact["train"]),
                "label_fields": sorted(label_artifact["train"]),
            }
            print(
                f"{name}: train={manifest['datasets'][name]['train_rows']}, "
                f"validation={manifest['datasets'][name]['validation_rows']}; "
                "inputs and labels saved separately",
                flush=True,
            )

            del split_data, preprocessor, input_artifact, label_artifact

        (TEMP_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        TEMP_DIR.replace(OUTPUT_DIR)
        print(f"STAGE6_TRAINVAL_ARTIFACTS_CREATED {OUTPUT_DIR}", flush=True)
    except Exception:
        # Keep a failed temporary build separate from the final artifact path.
        raise


if __name__ == "__main__":
    main()
