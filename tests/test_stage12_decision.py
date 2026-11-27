import numpy as np
import pandas as pd
import pytest

from novel_ids.decision.stage12 import DecisionThresholds, apply_decision_policy


def test_stage12_four_decision_states_and_precedence():
    classes = ["Normal", "DoS", "Probe_Recon"]
    probs = np.asarray([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.40, 0.35, 0.25],
        [0.80, 0.10, 0.10],
    ])
    scores = np.asarray([0.1, 0.1, 0.1, 2.0])
    out = apply_decision_policy(probs, scores, classes, DecisionThresholds(1.0, 0.6))
    assert out["decision"].tolist() == ["Normal", "Known attack", "Uncertain", "Unfamiliar"]
    assert out.loc[0, "predicted_category"] == "Normal"
    assert out.loc[1, "predicted_category"] == "DoS"
    assert pd.isna(out.loc[2, "predicted_category"])
    assert pd.isna(out.loc[3, "predicted_category"])


def test_unfamiliar_precedes_uncertain():
    out = apply_decision_policy(np.asarray([[0.45, 0.55]]), np.asarray([9.0]), ["Normal", "DoS"], DecisionThresholds(1.0, 0.9))
    assert out.loc[0, "decision"] == "Unfamiliar"


def test_rejects_nonfinite_or_malformed_probabilities():
    with pytest.raises(ValueError):
        apply_decision_policy(np.asarray([[np.nan, 0.0]]), np.asarray([0.0]), ["Normal", "DoS"], DecisionThresholds(1.0, 0.5))
    with pytest.raises(ValueError):
        apply_decision_policy(np.asarray([[0.2, 0.2]]), np.asarray([0.0]), ["Normal", "DoS"], DecisionThresholds(1.0, 0.5))

