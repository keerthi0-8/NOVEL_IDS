"""Evaluation report from saved prediction artifacts."""
import os
import csv
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"
PRED_DIR = os.path.join(
    PROJECT_ROOT, r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126\predictions"
)
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

NSL_CLASSES = ["Normal", "DoS", "Probe_Recon", "NSL_R2L", "NSL_U2R"]
UNSW_CLASSES = [
    "Normal", "UNSW_Generic", "UNSW_Exploits", "UNSW_Fuzzers",
    "Probe_Recon", "DoS", "UNSW_Backdoor", "UNSW_Analysis",
    "UNSW_Shellcode", "UNSW_Worms",
]
ALL_CLASSES = sorted(set(NSL_CLASSES) | set(UNSW_CLASSES))


def compute_metrics(df):
    mask = df["decision"].isin(["Normal", "Known_attack"])
    if mask.sum() == 0:
        return None
    y_true = df.loc[mask, "true_family"].values
    y_pred = df.loc[mask, "predicted_family"].values
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    per_class = {}
    for cls in classes:
        idx = class_to_idx[cls]
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "precision": prec, "recall": rec, "f1": f1, "support": int(support),
        }
    supported = [c for c in classes if per_class[c]["support"] > 0]
    macro_f1 = np.mean([per_class[c]["f1"] for c in supported]) if supported else 0.0
    total_support = sum(per_class[c]["support"] for c in supported)
    weighted_f1 = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in supported)
        / total_support if total_support > 0 else 0.0
    )
    weighted_prec = (
        sum(per_class[c]["precision"] * per_class[c]["support"] for c in supported)
        / total_support if total_support > 0 else 0.0
    )
    weighted_rec = (
        sum(per_class[c]["recall"] * per_class[c]["support"] for c in supported)
        / total_support if total_support > 0 else 0.0
    )
    accuracy = (y_true == y_pred).mean()
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "total": int(mask.sum()),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classes": classes,
        "total_support": int(total_support),
    }


def fmt_float(x):
    return f"{x:.4f}"


def build_report(dataset_name, metrics):
    lines = []
    lines.append("=" * 60)
    lines.append("NOVEL-IDS: MULTICLASS CLASSIFICATION EVALUATION  (" + dataset_name + ")")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. OVERALL PERFORMANCE")
    lines.append(
        "Accuracy: " + fmt_float(metrics["accuracy"])
        + " (" + str(int(metrics["accuracy"] * 10000) / 100) + "%)"
    )
    lines.append("Macro Precision: " + fmt_float(metrics["weighted_precision"]))
    lines.append("Macro Recall: " + fmt_float(metrics["weighted_recall"]))
    lines.append("Macro F1: " + fmt_float(metrics["macro_f1"]))
    lines.append("Weighted Precision: " + fmt_float(metrics["weighted_precision"]))
    lines.append("Weighted Recall: " + fmt_float(metrics["weighted_recall"]))
    lines.append("Weighted F1: " + fmt_float(metrics["weighted_f1"]))
    lines.append("Total Evaluated Samples: " + str(metrics["total"]))
    lines.append("")
    lines.append("2. CONFUSION MATRIX (rows = actual, columns = predicted)")
    classes = metrics["classes"]
    cm = metrics["confusion_matrix"]
    header = " " * 20 + " | ".join(f"{c:>12}" for c in classes)
    lines.append(header)
    lines.append("-" * len(header))
    for i, cls in enumerate(classes):
        row_vals = [f"{cm[i, j]:>12d}" for j in range(len(classes))]
        lines.append(f"{cls:<20} | " + " | ".join(row_vals))
    lines.append("")
    lines.append("3. PER-CLASS REPORT   Class | Precision | Recall | F1 | Support")
    for cls in classes:
        pc = metrics["per_class"][cls]
        lines.append(
            "  " + cls + " " * (24 - len(cls))
            + " " + f"{pc['precision']:.4f}"
            + " " + f"{pc['recall']:.4f}"
            + " " + f"{pc['f1']:.4f}"
            + " " + str(pc["support"])
        )
    lines.append("")
    lines.append("   Macro avg:  " + f"{metrics['macro_f1']:.4f}")
    lines.append("   Weighted avg: " + f"{metrics['weighted_f1']:.4f}")
    lines.append("   TOTAL SUPPORT: " + str(metrics["total_support"]))
    lines.append("")
    return lines


def main():
    nsl_path = os.path.join(PRED_DIR, "nsl_kdd_test_independent_primary.csv")
    unsw_path = os.path.join(PRED_DIR, "unsw_nb15_test_independent_primary.csv")
    nsl_df = pd.read_csv(nsl_path)
    unsw_df = pd.read_csv(unsw_path)
    print("NSL-KDD shape:", nsl_df.shape)
    print("UNSW-NB15 shape:", unsw_df.shape)
    print("Pooled shape:", nsl_df.shape[0] + unsw_df.shape[0])
    nsl_metrics = compute_metrics(nsl_df)
    unsw_metrics = compute_metrics(unsw_df)
    pooled_df = pd.concat([nsl_df, unsw_df], ignore_index=True)
    pooled_metrics = compute_metrics(pooled_df)
    print("=== NSL-KDD ===")
    print("  Accuracy:", f"{nsl_metrics['accuracy']:.4f}")
    print("  Macro F1:", f"{nsl_metrics['macro_f1']:.4f}")
    print("  Weighted F1:", f"{nsl_metrics['weighted_f1']:.4f}")
    print("  Total:", nsl_metrics["total"])
    print()
    print("=== UNSW-NB15 ===")
    print("  Accuracy:", f"{unsw_metrics['accuracy']:.4f}")
    print("  Macro F1:", f"{unsw_metrics['macro_f1']:.4f}")
    print("  Weighted F1:", f"{unsw_metrics['weighted_f1']:.4f}")
    print("  Total:", unsw_metrics["total"])
    print()
    print("=== POOLED ===")
    print("  Total:", pooled_metrics["total"])
    print("  Accuracy:", f"{pooled_metrics['accuracy']:.4f}")
    print("  Macro F1:", f"{pooled_metrics['macro_f1']:.4f}")
    print("  Weighted F1:", f"{pooled_metrics['weighted_f1']:.4f}")
    print()
    print("Expected: Total=95671, Accuracy=76.81%, Macro F1=47.67%, Weighted F1=77.68%")
    print("  Match total:", pooled_metrics["total"] == 95671)
    print("  Match acc:", abs(pooled_metrics["accuracy"] - 0.7681) < 0.001)
    print("  Match macro F1:", abs(pooled_metrics["macro_f1"] - 0.4767) < 0.001)
    print("  Match weighted F1:", abs(pooled_metrics["weighted_f1"] - 0.7768) < 0.001)
    all_lines = []
    all_lines.append("# NOVEL-IDS: MULTICLASS CLASSIFICATION EVALUATION")
    all_lines.append("")
    all_lines.append("Generated from saved prediction artifacts in:")
    all_lines.append("  " + PRED_DIR)
    all_lines.append("")
    all_lines.extend(build_report("NSL-KDD Independent Test", nsl_metrics))
    all_lines.append("")
    all_lines.extend(build_report("UNSW-NB15 Independent Test", unsw_metrics))
    all_lines.append("")
    all_lines.extend(build_report("Pooled (NSL-KDD + UNSW-NB15)", pooled_metrics))
    all_lines.append("")
    all_lines.append("6. How to read these numbers")
    all_lines.append("- support = number of true samples in a class")
    all_lines.append("- accuracy is dominated by large classes")
    all_lines.append("- macro metrics weigh all classes equally")
    all_lines.append("- weighted metrics weight each class by its support")
    all_lines.append("- NSL-KDD test has attack types not in training")
    all_lines.append("")
    all_lines.append("- A class with zero support prints '-'")
    all_lines.append("- A class with support but no correct prediction prints its real 0.00%")
    all_lines.append("")
    all_lines.append("Weak classes (NSL-R2L, NSL-U2R, UNSW_Analysis, UNSW_Backdoor,")
    all_lines.append("UNSW_Worms, UNSW_Fuzzers, UNSW_Shellcode) show low recall,")
    all_lines.append("indicating the model rarely detects rare attack types.")
    report_path = os.path.join(DOCS_DIR, "EVALUATION_REPORT.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(all_lines))
    print("Report written to", report_path)
    for name, metrics in [("nsl_kdd", nsl_metrics), ("unsw_nb15", unsw_metrics), ("pooled", pooled_metrics)]:
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"])
        cm_path = os.path.join(DOCS_DIR, "confusion_matrix_" + name + ".csv")
        cm_df.to_csv(cm_path)
        print("Confusion matrix saved:", cm_path)


if __name__ == "__main__":
    main()