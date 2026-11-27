from __future__ import annotations

import gc
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from novel_ids.evaluation.stage11_metrics import classification_metrics
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from novel_ids.decision.stage12 import DecisionThresholds, apply_decision_policy
from novel_ids.onboarding.unseen_schema import build_unseen_model_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path(r"C:\NOVEL_IDS_runs")
STAGE5_SHA = "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
STAGE6_SHA = "1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
PRIMARY_SHA = "349505b6ebae528246d72b66785b28e296ecdf4ccc6d775244005ae3fd5147c0"
CONFIDENCE_VALUE = {"high": 1.0, "medium": 0.8, "low": 0.5, "unassessed": 0.0}


@dataclass
class NovelIDSModel:
    device: torch.device
    stage5: torch.nn.Module
    branches: torch.nn.ModuleDict
    variant: dict[str, Any]
    taxonomy: Any
    identity_encoder: torch.nn.Module
    source_descriptors: list[dict[str, Any]]
    matching_descriptors: list[dict[str, Any]]
    mahalanobis: torch.nn.Module
    ood_combiner: torch.nn.Module
    calibration: dict[str, Any]
    _model_info: dict[str, Any]
    resource_module: Any
    source_preprocessors: dict[str, Any]
    source_slot_maps: dict[str, dict[str, int]]

    def model_info(self) -> dict[str, Any]:
        """Return public model version, provenance, taxonomy and frozen thresholds."""
        return json.loads(json.dumps(self._model_info))


@dataclass
class OnboardingReport:
    input_path: str
    rows: int
    feature_names: list[str]
    profile: dict[str, Any]
    feature_matches: list[dict[str, Any]]
    compatibility: dict[str, Any]
    model_inputs: dict[str, Any]
    frame: pd.DataFrame
    labels: pd.Series | None = None
    label_column: str | None = None
    normalization_input: str | None = None
    normalization_rows: int | None = None
    calibration_frame: pd.DataFrame | None = None
    feature_metadata: dict[str, dict[str, Any]] | None = None
    feature_metadata_path: str | None = None
    normalization_mode: str = "option_a_target"
    source_dataset: str | None = None
    streaming: bool = False

    @property
    def config(self) -> dict[str, Any]:
        """Serializable, dataset-specific onboarding configuration."""
        return {
            "schema_version": 1,
            "input_path": self.input_path,
            "input_rows": int(self.rows),
            "feature_columns_in_order": [str(name) for name in self.feature_names],
            "feature_matches_with_confidence": self.feature_matches,
            "feature_metadata_path": self.feature_metadata_path,
            "normalization_mode": self.normalization_mode,
            "compatibility": self.compatibility,
            "normalization": ({
                "method": "source-fitted training medians, log transforms, robust scalers and categorical vocabularies",
                "fit_scope": "audited source training partition only",
                "preprocessor": self.source_dataset,
                "statistics": "stored source preprocessor; no uploaded-sample fitting",
            } if self.source_dataset else {
                "method": "Option A: target-only median/IQR",
                "fit_scope": "unlabeled target feature rows only",
                "dataset_rows_scanned": int(self.normalization_rows or self.rows),
                "calibration_rows_used": int(self.profile.get("unlabeled_calibration_sample_rows", 0)),
                "calibration_seed": int(self.profile.get("calibration_sample_seed", 20261126)),
                "external_unlabeled_input": self.normalization_input,
                "statistics": self.model_inputs.get("unseen_numeric_normalization"),
            }),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "rows": self.rows,
            "feature_names": self.feature_names,
            "profile": self.profile,
            "feature_matches": self.feature_matches,
            "compatibility": self.compatibility,
            "config": self.config,
            "unseen_numeric_normalization": self.model_inputs.get("unseen_numeric_normalization"),
            "inference_mode": "source-schema with training-fitted preprocessing" if self.source_dataset else "zero-shot with unsupervised normalization",
            "recognized_dataset": self.source_dataset,
            "source_schema_recognized": self.source_dataset is not None,
            "normalization_input": self.normalization_input,
            "normalization_rows": self.normalization_rows,
        }


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _registry() -> tuple[torch.nn.Module, list[dict[str, Any]], list[dict[str, Any]]]:
    base = PROJECT_ROOT / "data/processed/unified_representation"
    identity_path = base / "feature_identity_encoder.joblib"
    schema_path = base / "unified_feature_schema.yaml"
    matching_path = base / "training_feature_descriptors.yaml"
    if not identity_path.is_file() or not schema_path.is_file() or not matching_path.is_file():
        raise FileNotFoundError("Feature identity registry artifacts are missing from the project.")
    encoder = joblib.load(identity_path)
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    matching_schema = yaml.safe_load(matching_path.read_text(encoding="utf-8"))
    descriptors = schema.get("features", [])
    matching_descriptors = matching_schema.get("features", [])
    if len(descriptors) != 78:
        raise RuntimeError(f"Expected 78 source model slots; found {len(descriptors)}.")
    if len(matching_descriptors) != 83 or any(not item.get("review_status") for item in matching_descriptors):
        raise RuntimeError("Expected 83 source-training matching descriptors with review status.")
    if not bool(getattr(encoder, "fitted", False)):
        raise RuntimeError("The saved feature identity encoder is not fitted.")
    return encoder, descriptors, matching_descriptors




_SOURCE_LABEL_COLUMNS = {"label", "attack_cat", "class", "target", "outcome"}
_SOURCE_METADATA_COLUMNS = {"nsl_kdd": {"difficulty_level"}, "unsw_nb15": {"id"}}
_SOURCE_COLUMN_ALIASES = {
    "protocol type": "protocol_type", "src bytes": "src_bytes", "dst bytes": "dst_bytes",
    "source bytes": "sbytes", "destination bytes": "dbytes",
    "source packets": "spkts", "destination packets": "dpkts",
}

def _normalise_source_column(name: Any) -> str:
    value = str(name).replace("\ufeff", "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return _SOURCE_COLUMN_ALIASES.get(value, value).replace(" ", "_")

def _load_source_preprocessors() -> dict[str, Any]:
    base = PROJECT_ROOT / "data" / "processed" / "unified_representation"
    result = {}
    for dataset in ("nsl_kdd", "unsw_nb15"):
        path = base / f"{dataset}_preprocessor.joblib"
        if not path.is_file(): raise FileNotFoundError(f"Source-fitted preprocessor is missing: {path}")
        processor = joblib.load(path)
        if not bool(getattr(processor, "fitted", False)): raise RuntimeError(f"Stored source preprocessor is not fitted: {dataset}")
        result[dataset] = processor
    return result

def _build_source_slot_maps(matching_descriptors, source_descriptors, preprocessors):
    semantic_slots = {str(x.get("semantic_id")): i for i,x in enumerate(source_descriptors) if x.get("semantic_id")}
    specific_slots = {str(x.get("token_key")): i for i,x in enumerate(source_descriptors) if x.get("token_key")}
    maps = {}
    for dataset, processor in preprocessors.items():
        mapping = {}
        for item in matching_descriptors:
            if item.get("dataset") != dataset: continue
            name = str(item.get("original_feature_name", ""))
            slot = semantic_slots.get(str(item["semantic_id"])) if int(item.get("level",3)) == 1 and item.get("semantic_id") else specific_slots.get(f"{dataset}::{name}")
            if slot is not None:
                if name in mapping and mapping[name] != slot: raise RuntimeError(f"Ambiguous source slot for {dataset}.{name}")
                mapping[name] = slot
        missing = sorted(set(processor.feature_names)-set(mapping))
        if missing: raise RuntimeError(f"Source registry lacks slots for {dataset}: {missing[:8]}")
        if len({mapping[x] for x in processor.feature_names}) != len(processor.feature_names): raise RuntimeError(f"Duplicate source slot mapping for {dataset}")
        maps[dataset] = {x:mapping[x] for x in processor.feature_names}
    return maps

def _recognize_source_schema(columns, preprocessors):
    supplied = [_normalise_source_column(x) for x in columns]
    for dataset in ("nsl_kdd","unsw_nb15"):
        processor = preprocessors.get(dataset)
        if processor is None: continue
        expected = {_normalise_source_column(x):str(x) for x in processor.feature_names}
        ignored = _SOURCE_LABEL_COLUMNS | _SOURCE_METADATA_COLUMNS[dataset]
        actual={}; valid=True
        for raw,canonical in zip(columns,supplied):
            if canonical in ignored: continue
            if canonical not in expected or canonical in actual: valid=False; break
            actual[canonical]=str(raw)
        if valid and set(actual)==set(expected): return dataset,{expected[k]:actual[k] for k in expected}
    return None,{}

def _build_source_model_inputs(frame, dataset, model):
    processor=model.source_preprocessors[dataset]; slot_map=model.source_slot_maps[dataset]
    expected={_normalise_source_column(x):str(x) for x in processor.feature_names}
    selected={}
    for raw in frame.columns:
        key=_normalise_source_column(raw)
        if key in _SOURCE_LABEL_COLUMNS or key in _SOURCE_METADATA_COLUMNS[dataset]: continue
        if key not in expected or key in selected: raise ValueError(f"Recognized {dataset} source schema changed after onboarding: {raw}")
        selected[key]=str(raw)
    if set(selected)!=set(expected): raise ValueError(f"Recognized {dataset} source schema is missing required features.")
    ordered=pd.DataFrame(index=frame.index)
    for key,name in expected.items(): ordered[name]=frame[selected[key]]
    for name in processor.numeric_columns: ordered[name]=pd.to_numeric(ordered[name],errors="coerce")
    transformed=processor.transform(ordered)
    numeric=np.zeros((len(ordered),78),dtype=np.float32); categorical=np.zeros((len(ordered),78),dtype=np.int64)
    available=np.zeros((len(ordered),78),dtype=np.float32); missing=np.zeros((len(ordered),78),dtype=np.float32)
    num_mask=np.zeros(78,dtype=bool); cat_mask=np.zeros(78,dtype=bool)
    for name in processor.numeric_columns:
        slot=slot_map[name]; numeric[:,slot]=transformed["numeric_values"][name].to_numpy(dtype=np.float32)
        missing[:,slot]=transformed["numeric_missing_mask"][name].to_numpy(dtype=np.float32); available[:,slot]=1; num_mask[slot]=True
    for name in processor.categorical_columns:
        slot=slot_map[name]; categorical[:,slot]=transformed["categorical_values"][name].to_numpy(dtype=np.int64)
        missing[:,slot]=transformed["categorical_missing_mask"][name].to_numpy(dtype=np.float32); available[:,slot]=1; cat_mask[slot]=True
    if not np.isfinite(numeric).all(): raise ValueError(f"{dataset} source preprocessing produced non-finite numeric values.")
    return {"numeric_values":numeric,"categorical_values":categorical,"M_avail":available,"M_miss":missing,
        "numeric_token_mask":num_mask,"categorical_token_mask":cat_mask,"source_feature_indices":np.arange(78,dtype=np.int64),
        "feature_descriptors":model.source_descriptors,"source_dataset":dataset,"is_unseen":False,
        "onboarded_feature_names":list(processor.feature_names),
        "source_preprocessing":{"method":"stored DatasetPreprocessor.transform","fit_scope":"audited source training partition only","normalization_fit_on_upload":False}}

def load_model(checkpoint_dir: str | Path, device: str | torch.device | None = None) -> NovelIDSModel:
    """Load frozen Stage 5-11 checkpoints and source-training OOD references.

    No test partition is opened. The Mahalanobis reference is reconstructed from
    the approved source-training sample using the same seeded Phase-C procedure.
    CPU is used automatically when CUDA is unavailable.
    """
    checkpoint_path = Path(checkpoint_dir).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available() and device is None:
        selected_device = torch.device("cpu")
    else:
        selected_device = torch.device(device or "cuda")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; retry with device='cpu'.")
    torch.set_num_threads(2)
    os.chdir(PROJECT_ROOT)
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_stage7_11_experiment as runner
    import run_phaseC_one_shot as phasec
    from novel_ids.taxonomy.stage11 import load_taxonomy

    runner.DEVICE = selected_device
    # Keep Phase-C helper device aligned with the selected inference device.
    phasec.r.DEVICE = selected_device
    # Each load attempt needs its own scratch directory; Streamlit reruns can
    # call load_model repeatedly within the same process (same PID).
    runner.RUN_DIR = RUNS_ROOT / f"inference_loader_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    runner.RUN_DIR.mkdir(parents=True, exist_ok=False)
    runner.taxonomy = load_taxonomy(PROJECT_ROOT / "config/taxonomy_unified.yaml", purpose="evaluation")
    runner.verify_protected()
    if _sha(phasec.STAGE5) != STAGE5_SHA or _sha(phasec.STAGE6) != STAGE6_SHA:
        raise RuntimeError("Frozen Stage 5/6 checkpoint hash mismatch.")
    actual_primary_sha = _sha(checkpoint_path)
    if actual_primary_sha != PRIMARY_SHA:
        raise RuntimeError(f"Unexpected Stage 7-11 checkpoint hash: {actual_primary_sha}")

    helper, branches = runner.read_data()
    runner.helper = helper
    labels = runner.load_labels(runner.taxonomy)
    stage5_checkpoint = torch.load(phasec.STAGE5, map_location="cpu", weights_only=False)
    aug_cfg = stage5_checkpoint["config"]
    del stage5_checkpoint
    level, confidence, group_only, level3_mask = runner.make_metadata()
    spec = ("primary", checkpoint_path, "full", "unified")
    variant = phasec.init_variant(spec, helper, branches, labels, aug_cfg,
                                  level, confidence, group_only, level3_mask, False)
    # Runtime placement only: Stage 5 and Stage 6 must share the selected device.
    helper.stage5.to(selected_device)
    for branch_module in branches.values():
        branch_module.to(selected_device)
    for module in variant.values():
        if isinstance(module, torch.nn.Module):
            module.to(selected_device)
    calibration = json.loads(phasec.CAL.read_text(encoding="utf-8"))
    mahalanobis, combiner = phasec.fit_primary_ood(
        variant, helper, branches, labels, aug_cfg, level, confidence,
        group_only, level3_mask, calibration,
    )

    stage5_model = helper.stage5
    identity_encoder, source_descriptors, matching_descriptors = _registry()
    source_preprocessors = _load_source_preprocessors()
    source_slot_maps = _build_source_slot_maps(matching_descriptors, source_descriptors, source_preprocessors)
    identity_encoder = identity_encoder.to(selected_device).eval()
    for parameter in identity_encoder.parameters():
        parameter.requires_grad_(False)
    helper.stage5.feature_identity_encoder = identity_encoder
    helper.stage5.eval()
    for parameter in helper.stage5.parameters():
        parameter.requires_grad_(False)
    for dataset in runner.DATASETS:
        helper.datasets[dataset].clear()
    helper.datasets = {dataset: {} for dataset in runner.DATASETS}
    del labels, helper
    gc.collect()
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    run_config_path = checkpoint_path.parent / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
    info = {
        "version": "NOVEL-IDS-1.0",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": int(run_config.get("seed", 43)),
        "taxonomy": list(runner.taxonomy.classes),
        "thresholds": {
            "temperature": float(calibration["temperature"]),
            "tau_ood": float(calibration["tau_ood_validation_p95"]),
            "tau_conf": float(calibration["tau_conf_validation_p10"]),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_primary_sha,
        "stage5_sha256": STAGE5_SHA,
        "stage6_sha256": STAGE6_SHA,
        "calibration_sha256": _sha(phasec.CAL),
        "taxonomy_sha256": _sha(PROJECT_ROOT / "config/taxonomy_unified.yaml"),
        "device": str(selected_device),
        "temperature": float(calibration["temperature"]),
        "tau_ood": float(calibration["tau_ood_validation_p95"]),
        "tau_conf": float(calibration["tau_conf_validation_p10"]),
        "unseen_branch_policy": "highest Stage-8 compatibility C across frozen source residual branches",
    }
    return NovelIDSModel(
        device=selected_device, stage5=stage5_model,
        branches=branches, variant=variant, taxonomy=runner.taxonomy,
        identity_encoder=identity_encoder, source_descriptors=source_descriptors,
        matching_descriptors=matching_descriptors, mahalanobis=mahalanobis, ood_combiner=combiner, calibration=calibration,
        _model_info=info, resource_module=runner,
        source_preprocessors=source_preprocessors, source_slot_maps=source_slot_maps,
    )



def _stream_csv_sample(
    path: str | Path,
    *,
    label_column: str | None = None,
    reject_label_columns: bool = False,
    chunksize: int = 8192,
    sample_rows: int = 50000,
) -> tuple[list[str], pd.DataFrame, dict[str, Any], int]:
    """Scan a CSV in bounded chunks and keep a deterministic uniform row sample."""
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if chunksize < 1 or sample_rows < 1:
        raise ValueError("chunksize and sample_rows must be positive.")
    rng = np.random.default_rng(20261126)
    sample = None
    seen = 0
    columns = None
    numeric_columns = None
    missing_counts = None
    inf_count = 0
    all_missing_counts = None
    try:
        iterator = pd.read_csv(input_path, chunksize=chunksize)
        for chunk in iterator:
            if columns is None:
                if len(chunk.columns) == 0 or not chunk.columns.is_unique:
                    raise ValueError("Input CSV must have columns with unique names.")
                columns = list(chunk.columns)
                if label_column is not None and label_column not in columns:
                    raise ValueError(f"Label column '{label_column}' was not found.")
                if reject_label_columns and any(str(c).strip().lower() in {"label", "class", "attack", "target"} for c in columns):
                    raise ValueError("Normalization CSV must not contain outcome-label columns.")
                feature_columns = [c for c in columns if c != label_column]
                if not feature_columns:
                    raise ValueError("At least one feature column is required after excluding labels.")
                numeric_columns = [
                    c for c in feature_columns
                    if pd.api.types.is_numeric_dtype(chunk[c].dtype)
                ]
                missing_counts = {c: 0 for c in feature_columns}
                all_missing_counts = {c: 0 for c in feature_columns}
            if list(chunk.columns) != columns:
                raise ValueError("CSV column schema changed between chunks.")
            features = chunk.drop(columns=[label_column]) if label_column is not None else chunk.copy()
            for c in numeric_columns:
                features[c] = pd.to_numeric(features[c], errors="coerce")
            for c in feature_columns:
                values = features[c]
                missing_counts[c] += int(values.isna().sum())
                all_missing_counts[c] += int(values.isna().sum())
            if numeric_columns:
                numeric = features[numeric_columns].to_numpy(dtype=np.float64, copy=True)
                inf_count += int(np.isinf(numeric).sum())
            m = len(features)
            if m:
                take = min(sample_rows, m)
                chunk_sample = features.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
                if sample is None:
                    sample = chunk_sample.reset_index(drop=True)
                    seen = m
                else:
                    desired = min(sample_rows, seen + m)
                    take_old = int(rng.hypergeometric(seen, m, desired))
                    take_new = desired - take_old
                    old_part = sample.sample(n=take_old, random_state=int(rng.integers(0, 2**31 - 1))) if take_old else sample.iloc[:0]
                    new_part = chunk_sample.sample(n=take_new, random_state=int(rng.integers(0, 2**31 - 1))) if take_new else chunk_sample.iloc[:0]
                    sample = pd.concat([old_part, new_part], ignore_index=True)
                    seen += m
        if columns is None or seen == 0 or sample is None or sample.empty:
            raise ValueError("Input CSV is empty or contains no data rows.")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty.") from exc
    feature_columns = [c for c in columns if c != label_column]
    rows = int(seen)
    profile = {
        "rows": rows,
        "input_columns": list(columns),
        "columns": len(feature_columns),
        "numeric_columns": len(numeric_columns),
        "numeric_feature_names": [str(c) for c in numeric_columns],
        "categorical_columns": len(feature_columns) - len(numeric_columns),
        "all_missing_columns": [str(c) for c in feature_columns if all_missing_counts[c] == rows],
        "infinite_numeric_values": int(inf_count),
        "missing_fraction_by_column": {str(c): float(missing_counts[c] / rows) for c in feature_columns},
        "streaming": True,
        "chunk_rows": int(chunksize),
        "unlabeled_calibration_sample_rows": int(len(sample)),
        "calibration_sample_seed": 20261126,
    }
    return feature_columns, sample.reset_index(drop=True), profile, rows


def onboard_csv(
    path: str | Path,
    model: NovelIDSModel | None = None,
    label_column: str | None = None,
    normalization_path: str | Path | None = None,
    *,
    chunksize: int = 8192,
    calibration_rows: int = 50000,
    feature_metadata_path: str | Path | None = None,
    normalization_mode: str = "option_a_target",
) -> OnboardingReport:
    """Profile and harmonize a CSV with bounded-memory, unlabeled calibration."""
    input_path = Path(path).expanduser()
    feature_names, calibration_frame, profile, rows = _stream_csv_sample(
        input_path, label_column=label_column, chunksize=chunksize,
        sample_rows=calibration_rows,
    )
    if model is None:
        encoder, descriptors, matching_descriptors = _registry()
        source_preprocessors = _load_source_preprocessors()
        source_slot_maps = _build_source_slot_maps(matching_descriptors, descriptors, source_preprocessors)
    else:
        encoder, descriptors, matching_descriptors = model.identity_encoder, model.source_descriptors, model.matching_descriptors
        source_preprocessors = getattr(model, "source_preprocessors", {})
        source_slot_maps = getattr(model, "source_slot_maps", {})
    source_dataset, _ = _recognize_source_schema(feature_names, source_preprocessors)

    feature_metadata = None
    metadata_path_resolved = None
    if feature_metadata_path is not None:
        metadata_file = Path(feature_metadata_path).expanduser()
        metadata_doc = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        feature_metadata = metadata_doc.get("features", {})
        if not isinstance(feature_metadata, dict):
            raise ValueError("External feature metadata must contain a features mapping.")
        unknown_metadata = sorted(set(feature_metadata) - set(feature_names))
        if unknown_metadata:
            raise ValueError(f"External metadata contains columns not present in the input: {unknown_metadata[:8]}")
        metadata_path_resolved = str(metadata_file.resolve())

    normalization_input_resolved = None
    normalization_rows = None
    if normalization_path is not None:
        normalization_file = Path(normalization_path).expanduser()
        norm_columns, norm_sample, norm_profile, norm_rows = _stream_csv_sample(
            normalization_file, reject_label_columns=True, chunksize=chunksize,
            sample_rows=calibration_rows,
        )
        if norm_columns != feature_names:
            raise ValueError("Normalization CSV columns/order must match evaluation feature columns exactly.")
        calibration_frame = norm_sample
        normalization_input_resolved = str(normalization_file.resolve())
        normalization_rows = int(norm_rows)
        profile["normalization_sample_rows"] = int(len(norm_sample))
    else:
        normalization_rows = int(rows)

    if source_dataset is not None:
        if normalization_path is not None:
            raise ValueError("Recognized source schemas must use source-fitted preprocessing, not an uploaded normalization file.")
        if model is None:
            from types import SimpleNamespace
            source_model = SimpleNamespace(source_preprocessors=source_preprocessors, source_slot_maps=source_slot_maps, source_descriptors=descriptors)
        else:
            source_model = model
        inputs = _build_source_model_inputs(calibration_frame, source_dataset, source_model)
        normalization_input_resolved = None
        normalization_rows = None
        source_slots = [source_slot_maps[source_dataset][name] for name in source_preprocessors[source_dataset].feature_names]
        extension = [descriptors[index] for index in source_slots]
        level_counts = {str(level): sum(int(descriptors[index].get("level", 3)) == level for index in source_slots) for level in (1, 2, 3)}
        compatibility = {
            "available_features": int(len(source_slots)),
            "level_counts": level_counts,
            "mean_match_confidence": 1.0,
            "level1_promotion_allowed": False,
            "source_slots_unavailable": 0,
            "recognized_source_schema": True,
            "recognized_dataset": source_dataset,
            "compatibility_score": 1.0,
            "trust_level": "High",
        }
    else:
        inputs = build_unseen_model_inputs(
            calibration_frame, descriptors, encoder,
            normalization_frame=calibration_frame,
            matching_descriptors=matching_descriptors,
            external_metadata=feature_metadata,
            similarity_threshold=0.15 if feature_metadata is not None else 1.0,
            normalization_mode=normalization_mode,
        )
        extension = inputs["feature_descriptors"][78:]
        level_counts = {str(level): sum(int(d.get("level", 3)) == level for d in extension) for level in (1, 2, 3)}
        confidence_values = [CONFIDENCE_VALUE.get(str(d.get("confidence", "unassessed")).lower(), 0.0) for d in extension]
        compatibility = {
            "available_features": int(len(extension)),
            "level_counts": level_counts,
            "mean_match_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
            "level1_promotion_allowed": bool(any(item.get("level") == 1 for item in extension)),
            "source_slots_unavailable": 78,
        }
    return OnboardingReport(
        input_path=str(input_path.resolve()), rows=rows,
        feature_names=list(feature_names), profile=profile,
        feature_matches=extension, compatibility=compatibility,
        model_inputs=inputs, frame=pd.DataFrame(), labels=None,
        label_column=label_column, feature_metadata=feature_metadata,
        feature_metadata_path=metadata_path_resolved,
        normalization_mode="source_fitted" if source_dataset else normalization_mode,
        normalization_input=normalization_input_resolved,
        normalization_rows=normalization_rows,
        calibration_frame=None if source_dataset else calibration_frame,
        streaming=True,
        source_dataset=source_dataset,
    )


def _predict_input_batch(
    model: NovelIDSModel,
    inputs: dict[str, Any],
    batch_size: int,
    row_offset: int,
) -> tuple[pd.DataFrame, list[str]]:
    if model.resource_module.free_ram_gb() < 2.0:
        raise RuntimeError("HARD STOP: free RAM below 2 GiB during inference.")
    descriptors = inputs["feature_descriptors"]
    descriptor_key = tuple(
        (int(d.get("level", 3)), str(d.get("confidence", "unassessed")).lower(),
         str(d.get("relationship", "")))
        for d in descriptors
    )
    slot_key = tuple(np.asarray(inputs["source_feature_indices"], dtype=np.int64).tolist())
    cache_key = (str(model.device), descriptor_key, slot_key)
    static_cache = getattr(model, "_inference_static_cache", None)
    if static_cache is None:
        static_cache = {}
        model._inference_static_cache = static_cache
    cached = static_cache.get(cache_key)
    if cached is None:
        cached = {
            "indices": torch.as_tensor(inputs["source_feature_indices"], device=model.device, dtype=torch.long),
            "levels": torch.as_tensor([int(d.get("level", 3)) for d in descriptors], device=model.device, dtype=torch.long),
            "confidences": torch.as_tensor([CONFIDENCE_VALUE.get(str(d.get("confidence", "unassessed")).lower(), 0.0) for d in descriptors], device=model.device, dtype=torch.float32),
            "groups": torch.as_tensor([str(d.get("relationship", "")) in {"group_only", "partial"} for d in descriptors], device=model.device, dtype=torch.bool),
        }
        bias_path = PROJECT_ROOT / "config" / "class_prior_bias.json"
        class_bias = np.zeros(len(model.taxonomy.classes), dtype=np.float32)
        if bias_path.is_file():
            bias_payload = json.loads(bias_path.read_text(encoding="utf-8"))
            expected_bias_sha = bias_payload.pop("bias_sha256", None)
            actual_bias_sha = hashlib.sha256(
                json.dumps(bias_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if expected_bias_sha != actual_bias_sha:
                raise RuntimeError("Configured class-prior bias SHA256 mismatch.")
            if bias_payload.get("classes") != list(model.taxonomy.classes):
                raise RuntimeError("Configured class-prior bias taxonomy order mismatch.")
            class_bias = np.asarray(
                [bias_payload["bias"][name] for name in model.taxonomy.classes], dtype=np.float32
            )
        cached["bias_tensor"] = torch.as_tensor(class_bias, device=model.device, dtype=torch.float32)
        static_cache.clear()
        static_cache[cache_key] = cached
    indices = cached["indices"]
    levels = cached["levels"]
    confidences = cached["confidences"]
    groups = cached["groups"]
    bias_tensor = cached["bias_tensor"]
    from novel_ids.alignment.stage8 import UNSEEN
    from novel_ids.ood.stage11 import energy_score

    n = int(inputs["numeric_values"].shape[0])
    tau = float(model.calibration["temperature"])
    all_probabilities, all_ood, all_branches = [], [], []
    for start in range(0, n, batch_size):
        if model.resource_module.free_ram_gb() < 2.0:
            raise RuntimeError("HARD STOP: free RAM below 2 GiB during inference.")
        if model.device.type == "cuda" and torch.cuda.max_memory_allocated() > 6 * 1024**3:
            raise RuntimeError("HARD STOP: GPU allocation exceeded 6 GiB during inference.")
        stop = min(n, start + batch_size)
        sl = slice(start, stop)
        numeric = torch.as_tensor(inputs["numeric_values"][sl], device=model.device, dtype=torch.float32)
        categorical = torch.as_tensor(inputs["categorical_values"][sl], device=model.device, dtype=torch.long)
        available = torch.as_tensor(inputs["M_avail"][sl], device=model.device, dtype=torch.float32)
        missing = torch.as_tensor(inputs["M_miss"][sl], device=model.device, dtype=torch.float32)
        numeric_mask = torch.as_tensor(inputs["numeric_token_mask"], device=model.device, dtype=torch.bool)
        categorical_mask = torch.as_tensor(inputs["categorical_token_mask"], device=model.device, dtype=torch.bool)
        with torch.no_grad():
            source_dataset = inputs.get("source_dataset")
            encoded = model.stage5(
                numeric_values=numeric, categorical_values=categorical,
                M_avail=available, M_miss=missing,
                numeric_token_mask=numeric_mask, categorical_token_mask=categorical_mask,
                dataset=source_dataset,
                feature_descriptors=None if source_dataset else descriptors,
                source_feature_indices=indices, is_unseen=source_dataset is None,
            )
            zs, zd, tokens = encoded["Z_s"], encoded["Z_d"], encoded["token_embeddings"]
            from novel_ids.alignment.stage8 import UNSEEN, NSL_KDD, UNSW_NB15
            source_id_value = (NSL_KDD if source_dataset == "nsl_kdd" else UNSW_NB15) if source_dataset else UNSEEN
            source_ids = torch.full((stop - start,), source_id_value, device=model.device, dtype=torch.long)
            candidates = []
            branch_names = (source_dataset,) if source_dataset else ("nsl_kdd", "unsw_nb15")
            for branch_name in branch_names:
                residual = model.branches[branch_name](zs, zd)
                gate_out = model.variant["stage7"](zs, residual["R_proj"])
                alignment = model.variant["stage8"](
                    gate_out["Z_f"], zd, source_ids, available.bool(),
                    levels.expand(stop - start, -1),
                    confidences.expand(stop - start, -1),
                    groups.expand(stop - start, -1),
                )
                fused = model.variant["stage9"](
                    z_s=zs, r=residual["R"], r_proj=residual["R_proj"],
                    gate=gate_out["G"], a=alignment["A"], u=alignment["U"],
                    c=alignment["C"], z_f=gate_out["Z_f"],
                )
                transformed = model.variant["stage10"](tokens, fused["Z_u"], available)
                logits = model.variant["classifier"](transformed["Z_t"])
                probabilities = torch.softmax(logits.float() / tau + bias_tensor.unsqueeze(0), dim=1)
                score_cpu = model.ood_combiner.score(
                    model.mahalanobis.score(fused["Z_u"].detach().cpu()),
                    energy_score(logits.float().detach().cpu(), temperature=tau),
                )
                score = score_cpu.to(model.device)
                candidates.append((alignment["C"].squeeze(1), alignment["distance"].squeeze(1), probabilities, score, branch_name))
            compatibility = torch.stack([x[0] for x in candidates], dim=1)
            distances = torch.stack([x[1] for x in candidates], dim=1)
            best_c = compatibility.max(dim=1, keepdim=True).values
            tied = compatibility >= (best_c - 1e-7)
            choice = distances.masked_fill(~tied, float("inf")).argmin(dim=1)
            row = torch.arange(stop - start, device=model.device)
            probs = torch.stack([x[2] for x in candidates], dim=1)[row, choice]
            scores = torch.stack([x[3] for x in candidates], dim=1)[row, choice]
            branch_names = np.asarray([x[4] for x in candidates], dtype=object)[choice.cpu().numpy()]
        all_probabilities.append(probs.cpu().numpy())
        all_ood.append(scores.cpu().numpy())
        all_branches.extend(branch_names.tolist())

    probabilities = np.concatenate(all_probabilities, axis=0)
    ood_scores = np.concatenate(all_ood, axis=0)
    decisions = apply_decision_policy(
        probabilities, ood_scores, model.taxonomy.classes,
        DecisionThresholds(float(model.calibration["tau_ood_validation_p95"]),
                           float(model.calibration["tau_conf_validation_p10"])),
    )
    output = decisions.reset_index(drop=True)
    best_indices = probabilities.argmax(axis=1)
    class_names = np.asarray(model.taxonomy.classes, dtype=object)
    output["best_guess_class"] = class_names[best_indices]
    output["best_guess_probability"] = probabilities[np.arange(n), best_indices].astype(np.float32)
    output.insert(0, "row_index", np.arange(row_offset, row_offset + n, dtype=np.int64))
    output["source_branch"] = all_branches
    for index, class_name in enumerate(model.taxonomy.classes):
        output[f"prob_{class_name}"] = probabilities[:, index].astype(np.float32)
    return output, all_branches


def predict_csv(
    path: str | Path,
    model: NovelIDSModel,
    onboarding: OnboardingReport | None = None,
    label_column: str | None = None,
    batch_size: int = 256,
    *,
    chunk_size: int = 8192,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict from a CSV in bounded chunks; labels are excluded before inference."""
    if batch_size < 1 or chunk_size < 1:
        raise ValueError("batch_size and chunk_size must be positive.")
    input_path = Path(path).expanduser()
    report = onboarding or onboard_csv(input_path, model=model, label_column=label_column, chunksize=chunk_size)
    if str(input_path.resolve()) != report.input_path:
        raise ValueError("The supplied onboarding report belongs to a different input file.")
    active_label = report.label_column if label_column is None else label_column
    if active_label != report.label_column:
        raise ValueError("label_column must match the onboarding report.")
    model.stage5.eval()
    for module in (model.variant["stage7"], model.variant["stage8"], model.variant["stage10"], model.variant["classifier"]):
        module.eval()

    outputs_path = None
    branch_counts = {}
    offset = 0
    progress_started = time.monotonic()
    progress_last = progress_started
    # Prediction output is much wider than the input. Spool each completed
    # chunk to disk so a large run does not retain all output chunks in RAM.
    disk = shutil.disk_usage(input_path.parent)
    if disk.free < 10 * 1024**3:
        raise RuntimeError("HARD STOP: free disk space below 10 GiB before streaming prediction output.")
    output_temp = tempfile.TemporaryDirectory(prefix="novel_ids_prediction_spool_")
    outputs_path = Path(output_temp.name) / "predictions.csv"
    wrote_output_header = False
    numeric_columns = report.profile.get("numeric_feature_names", [])
    chunks_since_gc = 0
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            if list(chunk.columns) != report.profile.get("input_columns"):
                raise ValueError("Input CSV schema/order changed after onboarding.")
            feature_frame = chunk.drop(columns=[active_label]) if active_label is not None else chunk
            # Source preprocessors perform their own numeric coercion. Avoid
            # converting the same columns twice on the recognized fast path.
            if not report.source_dataset:
                for column in numeric_columns:
                    feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
            if report.source_dataset:
                inputs = _build_source_model_inputs(feature_frame, report.source_dataset, model)
            else:
                inputs = build_unseen_model_inputs(
                    feature_frame, model.source_descriptors, model.identity_encoder,
                    normalization_frame=report.calibration_frame,
                    matching_descriptors=model.matching_descriptors,
                    external_metadata=report.feature_metadata,
                    similarity_threshold=0.15 if report.feature_metadata is not None else 1.0,
                    normalization_mode=report.normalization_mode,
                )
            batch_output, branches = _predict_input_batch(model, inputs, batch_size, offset)
            batch_output.to_csv(
                outputs_path,
                mode="a" if wrote_output_header else "w",
                header=not wrote_output_header,
                index=False,
            )
            wrote_output_header = True
            processed_rows = len(chunk)
            for name, count in zip(*np.unique(np.asarray(branches, dtype=str), return_counts=True)):
                branch_counts[str(name)] = branch_counts.get(str(name), 0) + int(count)
            offset += processed_rows
            del batch_output, inputs, feature_frame, chunk
            chunks_since_gc += 1
            # CPython releases these chunk objects by reference counting. A
            # full cyclic-GC scan on every chunk is costly; run it periodically
            # or when memory pressure approaches the protected safety floor.
            if chunks_since_gc >= 8 or model.resource_module.free_ram_gb() < 3.0:
                gc.collect()
                chunks_since_gc = 0
            if progress_callback is not None:
                progress_callback(offset, report.rows)
            now = time.monotonic()
            if now - progress_last >= 60.0:
                elapsed = now - progress_started
                print(f"[NOVEL-IDS] inference_progress rows={offset}/{report.rows} elapsed_seconds={elapsed:.1f}", flush=True)
                progress_last = now
    except pd.errors.EmptyDataError as exc:
        output_temp.cleanup()
        raise ValueError("Input CSV is empty.") from exc
    except Exception:
        output_temp.cleanup()
        raise
    if offset != report.rows or not wrote_output_header:
        output_temp.cleanup()
        raise RuntimeError(f"Streaming inference processed {offset} rows; onboarding recorded {report.rows}.")
    if model.resource_module.free_ram_gb() < 2.0:
        output_temp.cleanup()
        raise RuntimeError("HARD STOP: free RAM below 2 GiB before loading prediction results.")
    try:
        output = pd.read_csv(outputs_path)
    finally:
        output_temp.cleanup()
    tau = float(model.calibration["temperature"])
    class_counts = {
        str(class_name): int((output["predicted_category"] == class_name).sum())
        for class_name in model.taxonomy.classes
    }
    match_rows = report.feature_matches or []
    level_counts = report.compatibility.get("level_counts", {})
    if report.source_dataset:
        matched_exact = int(report.compatibility.get("available_features", 0))
        matched_semantic = group_only = dataset_specific = 0
        compatibility_score = 1.0
    else:
        matched_exact = sum(1 for d in match_rows if int(d.get("level", 3)) == 1 and d.get("relationship") == "equivalent")
        matched_semantic = sum(1 for d in match_rows if int(d.get("level", 3)) == 1 and d.get("relationship") not in {"equivalent", "group_only"})
        group_only = sum(1 for d in match_rows if int(d.get("level", 3)) == 2 or d.get("relationship") == "group_only")
        dataset_specific = sum(1 for d in match_rows if int(d.get("level", 3)) == 3)
        compatibility_score = float((matched_exact + matched_semantic + 0.5 * group_only) / max(1, len(match_rows)))
    trust_policy = yaml.safe_load((PROJECT_ROOT / "config" / "inference_trust_policy.yaml").read_text(encoding="utf-8"))
    trust_cfg = trust_policy["trust_levels"]
    trust_level = "High" if compatibility_score >= float(trust_cfg["high_min_compatibility"]) else (
        "Medium" if compatibility_score >= float(trust_cfg["medium_min_compatibility"]) else "Low"
    )
    compatibility_block = {
        "matched_exact": int(matched_exact),
        "matched_semantic": int(matched_semantic),
        "group_only": int(group_only),
        "dataset_specific": int(dataset_specific),
        "compatibility_score": compatibility_score,
        "level_counts": level_counts,
        "score_definition": "Exact recognized source-schema fingerprint" if report.source_dataset else "(exact + semantic + 0.5*group_only) / max(1, inspected external features)",
        "recognized_source_schema": report.source_dataset is not None,
        "recognized_dataset": report.source_dataset,
    }
    summary = {
        "rows": int(offset),
        "recognized_dataset": report.source_dataset,
        "inference_mode": "source_fitted" if report.source_dataset else "zero_shot_unseen",
        "compatibility": compatibility_block,
        "trust_level": trust_level,
        "decision_counts": {str(k): int(v) for k, v in output["decision"].value_counts().items()},
        "class_counts": class_counts,
        "source_branch_counts": branch_counts,
        "thresholds": {"temperature": tau, "tau_ood": float(model.calibration["tau_ood_validation_p95"]), "tau_conf": float(model.calibration["tau_conf_validation_p10"])},
        "unseen_normalization": None if report.source_dataset else report.model_inputs.get("unseen_numeric_normalization"),
        "onboarding": report.to_dict(),
        "model": model.model_info(),
        "streaming": {"enabled": True, "chunk_size": int(chunk_size), "batch_size": int(batch_size)},
    }
    return output, summary


def evaluate_predictions(predictions: pd.DataFrame, labels: Sequence[Any], classes: Sequence[str], label_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Optional evaluation-only label mapping; labels never enter prediction logic."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    raw = np.asarray(labels).astype(str)
    if len(raw) != len(predictions):
        raise ValueError("Evaluation labels and predictions must have equal row counts.")
    mapping = label_map or {}
    truth = np.asarray([mapping.get(x, x) for x in raw], dtype=object)
    valid = np.isin(truth, list(classes))
    accepted = predictions["decision"].isin(["Normal", "Known attack"]).to_numpy()
    scored = valid & accepted
    y_true = truth[scored]
    y_pred = predictions.loc[scored, "predicted_category"].astype(str).to_numpy()

      # Actual binary / multiclass evaluation
       
    terminal_metrics = None

    if len(y_true):
        terminal_metrics = classification_metrics(
            y_true=[list(classes).index(str(x)) for x in y_true],
            y_pred=[list(classes).index(str(x)) for x in y_pred],
            class_names=list(classes),
        )

    per_class = {}
    if len(y_true):
        p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=list(classes), zero_division=0)
        per_class = {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, c in enumerate(classes)}
    return {
        "rows": int(len(raw)), "unsupported_label_rows": int((~valid).sum()),
        "accepted_rows_with_supported_labels": int(scored.sum()),
        "coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "accepted_accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "accepted_macro_f1": float(f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)) if len(y_true) else None,
        "per_class": per_class,
        "terminal_metrics": terminal_metrics,

    }

