from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionThresholds:
    """Validation-fitted Stage 12 thresholds; never fitted on inference rows."""
    tau_ood: float
    tau_conf: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau_ood) or not np.isfinite(self.tau_conf):
            raise ValueError("Decision thresholds must be finite.")
        if not 0.0 <= self.tau_conf <= 1.0:
            raise ValueError("tau_conf must be within [0, 1].")


def apply_decision_policy(
    probabilities: np.ndarray,
    ood_scores: np.ndarray,
    classes: Sequence[str],
    thresholds: DecisionThresholds,
    *,
    normal_label: str = "Normal",
) -> pd.DataFrame:
    """Apply the approved precedence: Unfamiliar > Uncertain > Normal/Known attack.

    OOD and confidence thresholds are supplied from frozen validation calibration.
    This function does not fit or adapt thresholds from the input batch.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    scores = np.asarray(ood_scores, dtype=np.float64)
    labels = tuple(str(x) for x in classes)
    if probs.ndim != 2 or probs.shape[1] != len(labels) or not labels:
        raise ValueError("probabilities must have shape [rows, number of classes].")
    if len(set(labels)) != len(labels) or normal_label not in labels:
        raise ValueError("classes must be unique and include the configured Normal label.")
    if scores.shape != (probs.shape[0],):
        raise ValueError("ood_scores must have shape [rows].")
    if not np.isfinite(probs).all() or not np.isfinite(scores).all():
        raise ValueError("Probabilities and OOD scores must be finite.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Probabilities must be within [0, 1].")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    indices = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), indices]
    predicted = np.asarray(labels, dtype=object)[indices]
    unfamiliar = scores > float(thresholds.tau_ood)
    uncertain = (confidence < float(thresholds.tau_conf)) & ~unfamiliar
    decision = np.where(unfamiliar, "Unfamiliar", np.where(uncertain, "Uncertain", np.where(predicted == normal_label, "Normal", "Known attack")))
    category = np.where(np.isin(decision, ["Unfamiliar", "Uncertain"]), None, predicted)
    message = np.where(
        decision == "Unfamiliar", "Traffic differs from the learned source distribution.",
        np.where(decision == "Uncertain", "Model confidence is below the validation threshold.",
                 np.where(decision == "Normal", "Traffic classified as normal.", "Known attack category predicted.")),
    )
    return pd.DataFrame({
        "decision": decision,
        "predicted_category": category,
        "confidence": confidence.astype(np.float32),
        "ood_score": scores.astype(np.float32),
        "message": message,
    })
