#!/usr/bin/env python
"""Stream CIC-DDoS2019 CSVs and create evaluation/calibration samples.

The evaluation sample is stratified by source file and raw label. A separate
global reservoir sample is selected uniformly across all rows without using
labels; it is saved without the label column and is used only for Option A
unsupervised normalization and feature-descriptor statistics.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path(r"G:\My Drive\CiC-DDos")
DEFAULT_OUTPUT = Path(r"C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126")
SEED = 20261126
PER_FILE_LABEL_CAP = 7500
MAX_ROWS = 300_000
CALIBRATION_ROWS = 50_000
DEFAULT_CHUNK_ROWS = 10_000
MISSING_LABEL = "__MISSING_LABEL__"


def log(message: str, log_path: Path) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + os.linesep)


def update_reservoir(old: pd.DataFrame | None, old_seen: int,
                     new: pd.DataFrame, cap: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    if old is None or old.empty:
        if len(new) <= cap:
            return new.copy()
        seed = int(rng.integers(0, 2**32 - 1))
        return new.sample(n=cap, random_state=seed).copy()
    combined = pd.concat([old, new], ignore_index=True)
    if len(combined) <= cap:
        return combined
    old_weight = old_seen / len(old)
    weights = np.concatenate([
        np.full(len(old), old_weight, dtype=np.float64),
        np.ones(len(new), dtype=np.float64),
    ])
    seed = int(rng.integers(0, 2**32 - 1))
    return combined.sample(n=cap, weights=weights, random_state=seed).copy()


def find_label_column(columns) -> str:
    matches = [c for c in columns if str(c).strip().lower() in {"label", "class", "attack"}]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one label column; found {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--per-file-label-cap", type=int, default=PER_FILE_LABEL_CAP)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--calibration-rows", type=int, default=CALIBRATION_ROWS)
    args = parser.parse_args()

    source, output = args.source.expanduser(), args.output.expanduser()
    if not source.is_dir():
        raise SystemExit(f"Source directory is unavailable: {source}")
    if min(args.chunk_rows, args.per_file_label_cap, args.max_rows, args.calibration_rows) < 1:
        raise SystemExit("Chunk size and sample limits must be positive.")

    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "cic_sample.csv"
    manifest_path = output / "cic_sample_manifest.csv"
    calibration_path = output / "cic_unlabeled_calibration.csv"
    calibration_manifest_path = output / "cic_unlabeled_calibration_manifest.csv"
    profile_path = output / "cic_data_profile.json"
    log_path = output / "sample_preparation.log"
    for path in (sample_path, manifest_path, calibration_path, calibration_manifest_path, profile_path):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing artifact: {path}")

    files = sorted(
        (p for p in source.rglob("*.csv") if p.is_file()),
        key=lambda p: str(p.relative_to(source)).casefold(),
    )
    if not files:
        raise SystemExit(f"No CSV files found under {source}")

    rng = np.random.default_rng(SEED)
    reservoirs: dict[tuple[str, str], pd.DataFrame] = {}
    seen_by_group: Counter = Counter()
    calibration_reservoir: pd.DataFrame | None = None
    calibration_seen = 0
    profiles = []
    expected_columns = None
    label_column = None

    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        log(f"START file={file_index}/{len(files)} path={relative} bytes={path.stat().st_size}", log_path)
        rows_seen = 0
        label_counts: Counter = Counter()
        missing_counts: Counter = Counter()
        inf_counts: Counter = Counter()
        dtypes = {}
        file_columns = None

        reader = pd.read_csv(path, chunksize=args.chunk_rows, low_memory=False)
        for chunk_index, chunk in enumerate(reader, start=1):
            if file_columns is None:
                file_columns = list(chunk.columns)
                current_label = find_label_column(file_columns)
                if label_column is None:
                    label_column = current_label
                    expected_columns = file_columns
                elif current_label != label_column:
                    raise RuntimeError(f"Label column mismatch in {relative}: {current_label}")
                if file_columns != expected_columns:
                    raise RuntimeError(f"Schema mismatch in {relative}; do not silently align columns.")
                dtypes = {str(c): str(chunk[c].dtype) for c in chunk.columns}
            if chunk.empty:
                continue

            start_row = rows_seen
            row_indices = np.arange(start_row, start_row + len(chunk), dtype=np.int64)

            # Option A calibration reservoir: uniform over all target rows,
            # independent of raw labels, and the label column is never stored.
            calibration_chunk = chunk.drop(columns=[label_column]).copy()
            calibration_chunk["__source_file"] = relative
            calibration_chunk["__source_row_index"] = row_indices
            calibration_reservoir = update_reservoir(
                calibration_reservoir, calibration_seen, calibration_chunk,
                args.calibration_rows, rng,
            )
            calibration_seen += len(chunk)

            # Labels are used only for the separate evaluation-sample design.
            labels = chunk[label_column].map(
                lambda value: MISSING_LABEL if pd.isna(value) else str(value)
            )
            label_counts.update(labels.value_counts(dropna=False).to_dict())
            missing_counts.update(chunk.isna().sum().astype(int).to_dict())
            numeric = chunk.select_dtypes(include=[np.number])
            if len(numeric.columns):
                values = numeric.to_numpy(dtype=np.float64, copy=False)
                inf_counts.update({"__all_numeric__": int(np.isinf(values).sum())})

            eval_chunk = chunk.copy()
            eval_chunk["__source_file"] = relative
            eval_chunk["__source_row_index"] = row_indices
            eval_chunk["__raw_label_key"] = labels.to_numpy()

            for raw_label, group in eval_chunk.groupby("__raw_label_key", sort=False, dropna=False):
                key = (relative, str(raw_label))
                old = reservoirs.get(key)
                old_seen = int(seen_by_group[key])
                reservoirs[key] = update_reservoir(
                    old, old_seen, group, args.per_file_label_cap, rng
                )
                seen_by_group[key] += len(group)

            rows_seen += len(chunk)
            if chunk_index == 1 or chunk_index % 50 == 0:
                log(f"PROGRESS path={relative} chunks={chunk_index} rows_read={rows_seen}", log_path)

        if file_columns is None:
            raise RuntimeError(f"Empty CSV or unreadable header: {relative}")
        profiles.append({
            "relative_path": relative,
            "bytes": int(path.stat().st_size),
            "rows_read": int(rows_seen),
            "rows_skipped": 0,
            "columns": file_columns,
            "column_count": len(file_columns),
            "dtypes_first_chunk": dtypes,
            "label_column": label_column,
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "missing_counts_by_column": {str(k): int(v) for k, v in missing_counts.items()},
            "infinite_numeric_values": int(inf_counts.get("__all_numeric__", 0)),
            "sample_counts_by_label": {},
        })
        log(f"COMPLETE file={file_index}/{len(files)} path={relative} rows={rows_seen} labels={dict(label_counts)}", log_path)

    group_count = len(reservoirs)
    if group_count == 0 or calibration_reservoir is None:
        raise RuntimeError("No rows were sampled.")
    group_cap = min(args.per_file_label_cap, args.max_rows // group_count)
    if group_cap < 1:
        raise RuntimeError(f"Too many file/label groups ({group_count}) for max_rows={args.max_rows}.")

    sampled_groups = []
    for key in sorted(reservoirs):
        frame = reservoirs[key]
        if len(frame) > group_cap:
            seed = int(rng.integers(0, 2**32 - 1))
            frame = frame.sample(n=group_cap, random_state=seed)
        sampled_groups.append(frame)

    sampled = pd.concat(sampled_groups, ignore_index=True)
    sampled = sampled.sort_values(
        ["__source_file", "__raw_label_key", "__source_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(sampled) > args.max_rows:
        raise RuntimeError(f"Internal cap error: sampled {len(sampled)} > {args.max_rows}")

    manifest = pd.DataFrame({
        "sample_index": np.arange(len(sampled), dtype=np.int64),
        "source_file": sampled["__source_file"].astype(str),
        "source_row_index": sampled["__source_row_index"].astype(np.int64),
        "raw_label": sampled["__raw_label_key"].astype(str),
        "sampling_seed": SEED,
    })
    sample_frame = sampled.drop(columns=["__source_file", "__source_row_index", "__raw_label_key"])
    sample_frame.to_csv(sample_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    calibration = calibration_reservoir.sort_values(
        ["__source_file", "__source_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    calibration_manifest = pd.DataFrame({
        "calibration_index": np.arange(len(calibration), dtype=np.int64),
        "source_file": calibration["__source_file"].astype(str),
        "source_row_index": calibration["__source_row_index"].astype(np.int64),
        "sampling_seed": SEED,
    })
    calibration_frame = calibration.drop(columns=["__source_file", "__source_row_index"])
    calibration_frame.to_csv(calibration_path, index=False)
    calibration_manifest.to_csv(calibration_manifest_path, index=False)

    sample_counts = manifest.groupby(["source_file", "raw_label"]).size().to_dict()
    for profile in profiles:
        rel = profile["relative_path"]
        profile["sample_counts_by_label"] = {
            str(label): int(count)
            for (source_file, label), count in sample_counts.items()
            if source_file == rel
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "output_directory": str(output),
        "source_file_count": len(files),
        "total_rows_read": int(sum(p["rows_read"] for p in profiles)),
        "total_sample_rows": int(len(sampled)),
        "sample_cap": int(args.max_rows),
        "per_file_label_reservoir_cap": int(args.per_file_label_cap),
        "effective_group_cap": int(group_cap),
        "chunk_rows": int(args.chunk_rows),
        "sampling_seed": SEED,
        "sampling_policy": "uniform reservoir within each source-file/raw-label group; capped equally per group",
        "label_column": label_column,
        "labels_used_for": "evaluation sample stratification only; excluded from inference features and Option A calibration",
        "calibration_sample_rows": int(len(calibration_frame)),
        "calibration_sample_cap": int(args.calibration_rows),
        "calibration_sampling_policy": "single uniform reservoir across all target rows, independent of labels and source-file class proportions",
        "calibration_label_excluded": True,
        "normalization_policy": "Option A median/IQR and feature descriptors fitted only on the separate unlabeled calibration sample",
        "sha256_status": "deferred; full-file hashing was too slow on the streamed Google Drive mount",
        "source_files_modified": False,
        "files": profiles,
        "sample_csv": str(sample_path),
        "sample_manifest": str(manifest_path),
        "calibration_csv": str(calibration_path),
        "calibration_manifest": str(calibration_manifest_path),
    }
    profile_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")
    log(f"ALL COMPLETE files={len(files)} rows_read={summary['total_rows_read']} eval_sample={len(sampled)} calibration_sample={len(calibration_frame)}", log_path)
    print(json.dumps({
        "source_file_count": len(files),
        "total_rows_read": summary["total_rows_read"],
        "total_sample_rows": len(sampled),
        "calibration_sample_rows": len(calibration_frame),
        "sample_cap": args.max_rows,
        "group_count": group_count,
        "effective_group_cap": group_cap,
        "sample_csv": str(sample_path),
        "sample_manifest": str(manifest_path),
        "calibration_csv": str(calibration_path),
        "calibration_manifest": str(calibration_manifest_path),
        "profile": str(profile_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
