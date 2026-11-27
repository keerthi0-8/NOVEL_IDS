
"""
NOVEL-IDS | Combined Phase C Evaluation

Reads saved NSL-KDD and UNSW-NB15 confusion matrices,
combines them, and prints one pooled evaluation report.

No model training or inference is performed.
"""

import json
from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json"
)

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe_Recon",
    "UNSW_Exploits",
    "UNSW_Fuzzers",
    "UNSW_Generic",
    "UNSW_Analysis",
    "UNSW_Backdoor",
    "UNSW_Shellcode",
    "UNSW_Worms",
    "NSL_R2L",
    "NSL_U2R",
]


def evaluate_combined():

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    nsl_cm = np.array(
        results["datasets"]["nsl_kdd"]
        ["test_independent"]["primary"]["confusion_matrix"]
    )

    unsw_cm = np.array(
        results["datasets"]["unsw_nb15"]
        ["test_independent"]["primary"]["confusion_matrix"]
    )

    if nsl_cm.shape != (12, 12) or unsw_cm.shape != (12, 12):
        raise ValueError("Expected two 12x12 confusion matrices.")

    # Pool both datasets' saved test results.
    cm = nsl_cm + unsw_cm

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)

    fp = predicted - tp

    precision = np.divide(
        tp, predicted,
        out=np.zeros(12),
        where=predicted != 0
    )

    recall = np.divide(
        tp, support,
        out=np.zeros(12),
        where=support != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(12),
        where=(precision + recall) != 0
    )

    total = int(support.sum())

    if total == 0:
        raise ValueError("No test samples found.")

    active = support > 0
    weights = support / total

    accuracy = tp.sum() / total

    macro_precision = precision[active].mean()
    macro_recall = recall[active].mean()
    macro_f1 = f1[active].mean()

    weighted_precision = (precision * weights).sum()
    weighted_recall = (recall * weights).sum()
    weighted_f1 = (f1 * weights).sum()

    # Terminal report
    print("\n========== NOVEL-IDS:MULTI CLASS EVALUATION ==========")

    print(f"Total test samples : {total:,}")
    print(f"Classes evaluated : {active.sum()}/12")
    print(f"Accuracy           : {accuracy:.2%}")
    print(f"Macro Precision    : {macro_precision:.2%}")
    print(f"Macro Recall       : {macro_recall:.2%}")
    print(f"Macro F1           : {macro_f1:.2%}")
    print(f"Weighted Precision : {weighted_precision:.2%}")
    print(f"Weighted Recall    : {weighted_recall:.2%}")
    print(f"Weighted F1        : {weighted_f1:.2%}")

    print("\nCOMBINED PER-CLASS RESULTS")

    print(
        f"{'Class':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    for i, name in enumerate(CLASS_NAMES):

        if support[i] == 0:
            print(
                f"{name:<20}"
                f"{'—':>12}{'—':>12}{'—':>12}{0:>12}"
            )
        else:
            print(
                f"{name:<20}"
                f"{precision[i]:>11.2%}"
                f"{recall[i]:>12.2%}"
                f"{f1[i]:>12.2%}"
                f"{int(support[i]):>12,}"
            )

    print(
        "\nSource: saved Phase C NSL-KDD + UNSW-NB15 results."
    )
    print("No retraining or inference performed.")
    print("=====================================================\n")

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    evaluate_combined()
