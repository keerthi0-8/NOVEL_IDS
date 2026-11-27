#!/usr/bin/env python
"""Evaluation-only metrics for frozen NOVEL-IDS CIC-DDoS2019 predictions."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

OUT = Path(r"C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126")
PREDICTIONS = OUT / "cic_predictions_optionA.csv"
SUMMARY = OUT / "cic_inference_summary_optionA.json"
MANIFEST = OUT / "cic_sample_manifest.csv"
PROFILE = OUT / "cic_data_profile.json"
JOINED = OUT / "cic_predictions_with_manifest.csv"
METRICS = OUT / "cic_evaluation_metrics.json"
PER_FILE = OUT / "cic_per_file_metrics.csv"
PER_LABEL = OUT / "cic_per_label_metrics.csv"
MAPPING = OUT / "cic_label_mapping.json"

DOS_LABELS = {
    "drdos_dns", "drdos_ldap", "drdos_mssql", "drdos_netbios",
    "drdos_ntp", "drdos_ssdp", "drdos_udp",
    "dns", "ldap", "mssql", "netbios", "ntp", "ssdp",
    "syn", "tftp", "udp", "udp-lag", "udplag", "portmap",
    "ddos", "dos",
}
MISSING_LABEL = "__MISSING_LABEL__"


def map_label(raw: str) -> tuple[str, str]:
    value = str(raw).strip()
    key = value.casefold()
    if key == "benign":
        return "Normal", "approved_exact_normal"
    if key == "webddos":
        return "DoS", "approved_partial_application_layer_match"
    if key in DOS_LABELS:
        return "DoS", "approved_family_level_dos"
    return f"UNSUPPORTED_{value}", "unsupported_unmapped_label"


def safe_metrics(y_true, y_pred, labels):
    if len(y_true) == 0:
        return {"rows": 0, "accuracy": None, "macro_f1": None,
                "macro_f1_supported_classes": None, "weighted_f1": None,
                "per_class": {}, "confusion_matrix": None,
                "normalized_confusion_matrix": None}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0
    norm = np.divide(
        cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) > 0,
    )
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_supported_classes": float(np.mean(f1[present])) if present.any() else None,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": norm.tolist(),
    }


def ood_metrics(y_positive, scores):
    y = np.asarray(y_positive, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"defined": False, "reason": "OOD ground truth lacks both positive and negative samples."}
    fpr, tpr, _ = roc_curve(y, s)
    ix = np.flatnonzero(tpr >= 0.95)
    return {
        "defined": True,
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "fpr_at_95_tpr": float(fpr[ix[0]]) if len(ix) else 1.0,
        "positive_rows": int(y.sum()),
        "negative_rows": int((1-y).sum()),
    }


def evaluate(frame: pd.DataFrame) -> dict:
    raw = frame["raw_label"].astype(str)
    mapped = []
    statuses = []
    for value in raw:
        label, status = map_label(value)
        mapped.append(label)
        statuses.append(status)
    frame = frame.copy()
    frame["mapped_label"] = mapped
    frame["mapping_status"] = statuses

    decision = frame["decision"].astype(str)
    accepted = decision.isin(["Normal", "Known attack"]).to_numpy()
    pred_category = frame["predicted_category"].fillna("").astype(str).to_numpy()
    true_normal = np.asarray([x == "Normal" for x in mapped], dtype=bool)
    true_dos = np.asarray([x == "DoS" for x in mapped], dtype=bool)
    supported = true_normal | true_dos
    raw_present = raw.to_numpy() != MISSING_LABEL
    binary_valid = raw_present
    binary_truth = np.where(true_normal, 0, 1)
    binary_pred = np.where(pred_category == "Normal", 0, 1)
    binary_scored = binary_valid & accepted

    # Primary binary Attack-vs-Normal result is calculated on accepted outputs;
    # coverage and abstention/unfamiliar decisions are reported separately.
    binary = safe_metrics(
        binary_truth[binary_scored], binary_pred[binary_scored], [0, 1]
    )
    binary["class_names"] = ["Normal", "Attack"]
    binary["accepted_rows"] = int(binary_scored.sum())
    binary["total_nonmissing_label_rows"] = int(binary_valid.sum())
    binary["accepted_coverage"] = float(binary_scored.sum()/binary_valid.sum()) if binary_valid.any() else 0.0

    known_scored = supported & accepted
    known_truth = np.where(true_normal, "Normal", "DoS")
    known_pred = np.where(pred_category == "Normal", "Normal", "DoS")
    known = safe_metrics(known_truth[known_scored].tolist(), known_pred[known_scored].tolist(), ["Normal", "DoS"])
    known["accepted_rows"] = int(known_scored.sum())
    known["supported_label_rows"] = int(supported.sum())
    known["accepted_coverage"] = float(known_scored.sum()/supported.sum()) if supported.any() else 0.0

    unsupported = np.asarray([s == "unsupported_unmapped_label" for s in statuses], dtype=bool) & raw_present
    ood = ood_metrics(unsupported, frame["ood_score"].to_numpy())

    detection_positive = decision.isin(["Known attack", "Unfamiliar"]).to_numpy()
    decision_counts = {str(k): int(v) for k, v in decision.value_counts().items()}
    return {
        "rows": int(len(frame)),
        "supported_rows": int(supported.sum()),
        "unsupported_rows": int(unsupported.sum()),
        "missing_label_rows": int((~raw_present).sum()),
        "decision_counts": decision_counts,
        "binary_attack_vs_normal_accepted": binary,
        "known_normal_vs_dos_accepted": known,
        "ood_unsupported_label_detection": ood,
        "attack_detection_rate_on_nonbenign": (
            float(detection_positive[~true_normal & raw_present].mean())
            if np.any(~true_normal & raw_present) else None
        ),
        "uncertain_rate": float((decision == "Uncertain").mean()) if len(frame) else 0.0,
        "unfamiliar_rate": float((decision == "Unfamiliar").mean()) if len(frame) else 0.0,
        "mapping_counts": {str(k): int(v) for k, v in frame["mapping_status"].value_counts().items()},
    }


def main():
    for path in (PREDICTIONS, SUMMARY, MANIFEST, PROFILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase D artifact missing: {path}")
    # JOINED may already have been emitted by the inference workflow. Reuse it
    # only after exact validation against the immutable manifest + predictions.
    for path in (METRICS, PER_FILE, PER_LABEL, MAPPING):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite one-shot evaluation artifact: {path}")

    predictions = pd.read_csv(PREDICTIONS)
    manifest = pd.read_csv(MANIFEST)
    if len(predictions) != len(manifest):
        raise RuntimeError(f"Prediction/manifest row mismatch: {len(predictions)} vs {len(manifest)}")
    if not np.array_equal(predictions["row_index"].to_numpy(), manifest["sample_index"].to_numpy()):
        raise RuntimeError("Prediction row_index does not align with immutable sample manifest.")

    frame = pd.concat([manifest.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    frame = frame.rename(columns={"source_file": "source_filename"})
    frame["raw_label"] = frame["raw_label"].astype(str)
    if JOINED.exists():
        existing = pd.read_csv(JOINED)
        if not existing.equals(frame):
            raise RuntimeError("Existing predictions-with-manifest artifact does not exactly match immutable inputs.")
    else:
        frame.to_csv(JOINED, index=False)

    overall = evaluate(frame)
    per_file_rows = []
    for source_file, group in frame.groupby("source_filename", sort=True):
        metrics = evaluate(group.reset_index(drop=True))
        per_file_rows.append({
            "source_filename": source_file,
            "rows": metrics["rows"],
            "supported_rows": metrics["supported_rows"],
            "unsupported_rows": metrics["unsupported_rows"],
            "accepted_coverage": metrics["binary_attack_vs_normal_accepted"]["accepted_coverage"],
            "binary_accuracy_accepted": metrics["binary_attack_vs_normal_accepted"]["accuracy"],
            "binary_macro_f1_accepted": metrics["binary_attack_vs_normal_accepted"]["macro_f1_supported_classes"],
            "normal_dos_accuracy_accepted": metrics["known_normal_vs_dos_accepted"]["accuracy"],
            "normal_dos_macro_f1_accepted": metrics["known_normal_vs_dos_accepted"]["macro_f1_supported_classes"],
            "uncertain_rate": metrics["uncertain_rate"],
            "unfamiliar_rate": metrics["unfamiliar_rate"],
        })
    pd.DataFrame(per_file_rows).to_csv(PER_FILE, index=False)

    per_label_rows = []
    for raw_label, group in frame.groupby("raw_label", sort=True):
        mapped_label, status = map_label(raw_label)
        decisions = group["decision"].astype(str)
        per_label_rows.append({
            "raw_label": raw_label,
            "mapped_label": mapped_label,
            "mapping_status": status,
            "support": int(len(group)),
            "decision_normal": int((decisions == "Normal").sum()),
            "decision_known_attack": int((decisions == "Known attack").sum()),
            "decision_uncertain": int((decisions == "Uncertain").sum()),
            "decision_unfamiliar": int((decisions == "Unfamiliar").sum()),
            "attack_detected_rate": float(decisions.isin(["Known attack", "Unfamiliar"]).mean()),
            "mean_confidence": float(group["confidence"].mean()),
            "mean_ood_score": float(group["ood_score"].mean()),
        })
    pd.DataFrame(per_label_rows).to_csv(PER_LABEL, index=False)

    inference = json.loads(SUMMARY.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    feature_matches = inference.get("onboarding", {}).get("feature_matches", [])
    level_counts = {}
    for item in feature_matches:
        level = str(item.get("level", "unknown"))
        level_counts[level] = level_counts.get(level, 0) + 1

    label_map = {}
    for value in sorted(frame["raw_label"].astype(str).unique()):
        mapped, status = map_label(value)
        label_map[value] = {"mapped_label": mapped, "status": status}
    MAPPING.write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "one-shot, frozen zero-shot CIC-DDoS2019 evaluation; labels used only after prediction",
        "sample_rows": int(len(frame)),
        "source_file_count": int(frame["source_filename"].nunique()),
        "sample_seed": profile.get("sampling_seed"),
        "sample_cap": profile.get("sample_cap"),
        "source_rows_read": profile.get("total_rows_read"),
        "per_file": per_file_rows,
        "aggregate": overall,
        "feature_matching": {
            "new_feature_count": len(feature_matches),
            "level_counts": level_counts,
            "strict_option_b_exact_match_count": 0,
            "strict_option_b_status": "not applicable: no literal or whitespace-trimmed exact CIC-to-source feature-name matches were found; no source statistics were applied",
        },
        "option_a_normalization": inference.get("unseen_normalization"),
        "thresholds": inference.get("thresholds"),
        "source_branch_counts": inference.get("source_branch_counts"),
        "model": inference.get("model"),
        "coverage_warning": "Partial CIC-DDoS2019 coverage only. Exact absent filenames: NetBIOS.csv, TFTP.csv, UDP.csv. Do not generalize to absent files or attack families.",
        "artifacts": {
            "predictions_with_manifest": str(JOINED),
            "per_file_metrics": str(PER_FILE),
            "per_label_metrics": str(PER_LABEL),
            "label_mapping": str(MAPPING),
        },
    }
    METRICS.write_text(json.dumps(result, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")
    print(json.dumps({"rows": len(frame), "files": result["source_file_count"], "metrics": str(METRICS)}, indent=2))


if __name__ == "__main__":
    main()
