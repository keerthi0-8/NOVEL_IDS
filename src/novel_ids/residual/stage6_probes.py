from __future__ import annotations

from typing import Any

from novel_ids.residual.stage6_data import load_stage6_label_artifact

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic_probe_pair(
    *,
    train_z_s: np.ndarray,
    train_r: np.ndarray,
    train_labels: np.ndarray,
    validation_z_s: np.ndarray,
    validation_r: np.ndarray,
    validation_labels: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit paired native-label probes on train only; evaluate on source validation."""
    arrays = {
        "train_z_s": np.asarray(train_z_s, dtype=np.float32),
        "train_r": np.asarray(train_r, dtype=np.float32),
        "validation_z_s": np.asarray(validation_z_s, dtype=np.float32),
        "validation_r": np.asarray(validation_r, dtype=np.float32),
    }
    y_train = np.asarray(train_labels).astype(str)
    y_val = np.asarray(validation_labels).astype(str)
    if any(x.ndim != 2 for x in arrays.values()):
        raise ValueError("Probe feature matrices must be rank-2.")
    if arrays["train_z_s"].shape[0] != y_train.shape[0]:
        raise ValueError("Training features and labels row counts differ.")
    if arrays["train_r"].shape[0] != y_train.shape[0]:
        raise ValueError("Training residuals and labels row counts differ.")
    if arrays["validation_z_s"].shape[0] != y_val.shape[0]:
        raise ValueError("Validation features and labels row counts differ.")
    if arrays["validation_r"].shape[0] != y_val.shape[0]:
        raise ValueError("Validation residuals and labels row counts differ.")
    if arrays["train_z_s"].shape[1] != arrays["validation_z_s"].shape[1]:
        raise ValueError("Z_s train/validation dimensions differ.")
    if arrays["train_r"].shape[1] != arrays["validation_r"].shape[1]:
        raise ValueError("R train/validation dimensions differ.")
    if np.unique(y_train).size < 2:
        raise ValueError("Logistic probe requires at least two training classes.")
    if y_val.size == 0:
        raise ValueError("Validation labels are empty.")
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise FloatingPointError("Probe features contain NaN or Inf.")

    supported = np.unique(y_val)
    inputs = {
        "Z_s": (
            arrays["train_z_s"],
            arrays["validation_z_s"],
        ),
        "[Z_s,R]": (
            np.concatenate([arrays["train_z_s"], arrays["train_r"]], axis=1),
            np.concatenate([arrays["validation_z_s"], arrays["validation_r"]], axis=1),
        ),
    }
    results: dict[str, Any] = {}
    for label, (x_train, x_val) in inputs.items():
        # Scaling is fitted on training features only. Balanced class weights
        # are computed by sklearn from training labels only.
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        probe.fit(x_train, y_train)
        prediction = probe.predict(x_val)
        macro = float(f1_score(
            y_val, prediction, labels=supported, average="macro", zero_division=0
        ))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, prediction, labels=supported, zero_division=0
        )
        per_class = {
            str(class_name): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, class_name in enumerate(supported)
        }
        results[label] = {
            "validation_macro_f1": macro,
            "validation_class_support": {
                str(class_name): int(np.sum(y_val == class_name))
                for class_name in supported
            },
            "per_class": per_class,
            "train_rows": int(y_train.size),
            "validation_rows": int(y_val.size),
            "train_class_count": int(np.unique(y_train).size),
            "validation_supported_class_count": int(supported.size),
        }

    results["macro_f1_gain"] = (
        results["[Z_s,R]"]["validation_macro_f1"]
        - results["Z_s"]["validation_macro_f1"]
    )
    return results


def run_source_validation_probes(
    trainer, *, seed: int = 42
) -> dict[str, Any]:
    """Run paired probes using source train labels and source validation labels only."""
    label_paths = trainer.provenance.get("source_label_artifact_paths")
    if not label_paths or set(label_paths) != {"nsl_kdd", "unsw_nb15"}:
        raise ValueError("Probe evaluation requires separate train/validation label artifact provenance.")
    output: dict[str, Any] = {}
    gains = []
    for name in ("nsl_kdd", "unsw_nb15"):
        artifact = load_stage6_label_artifact(label_paths[name], name)
        y_train = np.asarray(artifact["train"]["labels"]).astype(str)
        y_val = np.asarray(artifact["validation"]["labels"]).astype(str)
        label_metadata = dict(artifact["metadata"])
        del artifact
        train_z_s, train_r = trainer.extract_features(name, "train")
        val_z_s, val_r = trainer.extract_features(name, "validation")
        result = fit_logistic_probe_pair(
            train_z_s=train_z_s,
            train_r=train_r,
            train_labels=y_train,
            validation_z_s=val_z_s,
            validation_r=val_r,
            validation_labels=y_val,
            seed=seed,
        )
        result["label_artifact_metadata"] = label_metadata
        output[name] = result
        gains.append(result["macro_f1_gain"])
    output["mean_probe_gain"] = float(np.mean(gains))
    output["probe_seed"] = int(seed)
    output["label_policy"] = (
        "native source labels; train labels fit probes only; validation labels score only"
    )
    return output
