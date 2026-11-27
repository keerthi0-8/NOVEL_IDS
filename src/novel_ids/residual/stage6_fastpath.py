from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_augmentation_masks(
    available: np.ndarray,
    missing: np.ndarray,
    numeric_token_mask: np.ndarray,
    categorical_token_mask: np.ndarray,
    level3_positions: Sequence[int],
    rng: np.random.Generator,
    feature_set_dropout_rate: float,
    feature_set_dropout_all_probability: float,
    value_masking_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the approved Stage5 feature-set/value masking without enabling NN dropout."""
    avail = np.asarray(available, dtype=np.float32).copy()
    miss = np.asarray(missing, dtype=np.float32).copy()
    batch, count = avail.shape
    specific = np.zeros(count, dtype=bool)
    specific[np.asarray(level3_positions, dtype=np.int64)] = True
    candidates = (avail > 0) & specific[None, :]
    feature_drop = (rng.random((batch, count)) < feature_set_dropout_rate) & candidates
    drop_all = rng.random((batch, 1)) < feature_set_dropout_all_probability
    feature_drop = np.where(drop_all, candidates, feature_drop)
    avail[feature_drop] = 0.0
    miss[feature_drop] = 0.0
    typed = (
        np.asarray(numeric_token_mask, dtype=bool)
        | np.asarray(categorical_token_mask, dtype=bool)
    )[None, :]
    eligible = (avail > 0) & (miss == 0) & typed
    value_mask = (rng.random((batch, count)) < value_masking_rate) & eligible
    miss[value_mask] = 1.0
    return avail, miss, feature_drop, value_mask


@torch.no_grad()
def stage5_encode_augmented(
    stage5: torch.nn.Module,
    numeric_values: np.ndarray,
    categorical_values: np.ndarray,
    available: np.ndarray,
    missing: np.ndarray,
    numeric_token_mask: np.ndarray,
    categorical_token_mask: np.ndarray,
    feature_drop_mask: np.ndarray,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen/eval Stage5 with external masks and the matching feature-drop pool mask.

    The existing Stage5 trainer passes feature_descriptors=None. To preserve that
    path while activating the approved Level-3 feature dropout, recompute only the
    final pooling using the exact dropout exclusion mask. All learned Stage5
    modules remain unchanged, frozen, eval-mode and under no_grad.
    """
    device = stage5.source_identity_base.device
    numeric = torch.as_tensor(numeric_values, device=device, dtype=torch.float32)
    categorical = torch.as_tensor(categorical_values, device=device, dtype=torch.long)
    avail = torch.as_tensor(available, device=device, dtype=torch.float32)
    miss = torch.as_tensor(missing, device=device, dtype=torch.float32)
    num_mask = torch.as_tensor(numeric_token_mask, device=device, dtype=torch.bool)
    cat_mask = torch.as_tensor(categorical_token_mask, device=device, dtype=torch.bool)
    dropped = torch.as_tensor(feature_drop_mask, device=device, dtype=torch.bool)

    stage5.eval()
    output = stage5(
        numeric_values=numeric,
        categorical_values=categorical,
        M_avail=avail,
        M_miss=miss,
        numeric_token_mask=num_mask,
        categorical_token_mask=cat_mask,
        dataset=dataset,
        feature_descriptors=None,
    )
    tokens = output["token_embeddings"]
    # All 78 slots are source slots. Keep structurally unavailable slots in the
    # pool as before, but exclude rows explicitly hidden by feature-set dropout.
    pool_valid = avail.bool() | (~dropped)
    weights = pool_valid.unsqueeze(-1).to(tokens.dtype)
    pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    context = output["feature_context"]
    z_s = stage5.shared_encoder(pooled)
    z_d = stage5.shared_context_adapter(pooled, context)
    return z_s.float(), z_d.float()


def macro_f1_supported(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    supported_labels: np.ndarray | None = None,
) -> float:
    """Macro-F1 over classes supported in the original validation set."""
    y = np.asarray(y_true).astype(str)
    pred = np.asarray(y_pred).astype(str)
    labels = np.unique(y) if supported_labels is None else np.asarray(supported_labels).astype(str)
    if labels.size == 0:
        raise ValueError("No validation-supported labels.")
    c = labels.size
    yt = np.searchsorted(labels, y)
    yp = np.searchsorted(labels, pred)
    valid = yp < c
    valid &= labels[np.minimum(yp, c - 1)] == pred
    keys = yt[valid].astype(np.int64) * c + yp[valid].astype(np.int64)
    cm = np.bincount(keys, minlength=c * c).reshape(c, c).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def paired_bootstrap_ci(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_aug: np.ndarray,
    seed: int = 42,
    repetitions: int = 2000,
    batch_size: int = 50,
) -> dict[str, float]:
    """Paired vectorized row bootstrap; each replicate uses identical rows for both probes."""
    y = np.asarray(y_true).astype(str)
    base = np.asarray(pred_base).astype(str)
    aug = np.asarray(pred_aug).astype(str)
    if not (y.size == base.size == aug.size):
        raise ValueError("Bootstrap arrays must have equal row counts.")
    labels = np.unique(y)
    observed = macro_f1_supported(y, aug, labels) - macro_f1_supported(y, base, labels)
    c = labels.size
    yt = np.searchsorted(labels, y).astype(np.int32)

    def pair_codes(pred: np.ndarray) -> np.ndarray:
        pi = np.searchsorted(labels, pred)
        valid = pi < c
        valid &= labels[np.minimum(pi, c - 1)] == pred
        codes = np.full(pred.size, -1, dtype=np.int32)
        codes[valid] = yt[valid] * c + pi[valid].astype(np.int32)
        return codes

    base_codes = pair_codes(base)
    aug_codes = pair_codes(aug)
    rng = np.random.default_rng(seed)
    gains = np.empty(repetitions, dtype=np.float64)

    def score_sampled(codes: np.ndarray, indices: np.ndarray) -> np.ndarray:
        b, n = indices.shape
        sampled = codes[indices]
        valid = sampled >= 0
        offsets = (np.arange(b, dtype=np.int64)[:, None] * (c * c))
        keys = sampled[valid].astype(np.int64) + np.broadcast_to(offsets, sampled.shape)[valid]
        cm = np.bincount(keys, minlength=b * c * c).reshape(b, c, c).astype(np.float64)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        fp = cm.sum(axis=1) - tp
        fn = cm.sum(axis=2) - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        return f1.mean(axis=1)

    for start in range(0, repetitions, batch_size):
        stop = min(repetitions, start + batch_size)
        indices = rng.integers(0, y.size, size=(stop - start, y.size), dtype=np.int32)
        gains[start:stop] = (
            score_sampled(aug_codes, indices) - score_sampled(base_codes, indices)
        )
    return {
        "observed_gain": float(observed),
        "ci95_lower": float(np.quantile(gains, 0.025)),
        "ci95_upper": float(np.quantile(gains, 0.975)),
        "bootstrap_replicates": int(repetitions),
        "bootstrap_seed": int(seed),
    }


def verify_bootstrap_against_sklearn(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_aug: np.ndarray,
    seed: int = 42,
    repetitions: int = 50,
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    y = np.asarray(y_true).astype(str)
    base = np.asarray(pred_base).astype(str)
    aug = np.asarray(pred_aug).astype(str)
    labels = np.unique(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repetitions):
        idx = rng.integers(0, y.size, size=y.size)
        expected = (
            f1_score(y[idx], aug[idx], labels=labels, average="macro", zero_division=0)
            - f1_score(y[idx], base[idx], labels=labels, average="macro", zero_division=0)
        )
        actual = (
            macro_f1_supported(y[idx], aug[idx], labels)
            - macro_f1_supported(y[idx], base[idx], labels)
        )
        diffs.append(abs(float(expected) - float(actual)))
    maximum = float(max(diffs, default=0.0))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "max_abs_difference": maximum,
        "tolerance": 1e-12,
        "passed": bool(maximum <= 1e-12),
    }


def standardize_feature_blocks(
    train_blocks: Mapping[str, np.ndarray],
    validation_blocks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one scaler per feature block using training rows only."""
    names = list(train_blocks)
    if names != list(validation_blocks):
        raise ValueError("Train/validation feature-block names or order differ.")
    train_parts = []
    val_parts = []
    stats: dict[str, Any] = {}
    for name in names:
        tr = np.asarray(train_blocks[name], dtype=np.float32)
        va = np.asarray(validation_blocks[name], dtype=np.float32)
        if tr.ndim != 2 or va.ndim != 2 or tr.shape[1] != va.shape[1]:
            raise ValueError(f"Invalid feature block shape for {name}.")
        mean = tr.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = tr.std(axis=0, dtype=np.float64).astype(np.float32)
        safe_std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        train_parts.append((tr - mean) / safe_std)
        val_parts.append((va - mean) / safe_std)
        stats[name] = {
            "mean": mean.tolist(),
            "scale": safe_std.tolist(),
            "train_rows": int(tr.shape[0]),
        }
    return (
        np.concatenate(train_parts, axis=1).astype(np.float32, copy=False),
        np.concatenate(val_parts, axis=1).astype(np.float32, copy=False),
        stats,
    )


def fit_logistic_probe(
    train_blocks: Mapping[str, np.ndarray],
    validation_blocks: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int = 42,
    standardize_blocks: bool = True,
) -> dict[str, Any]:
    """Fit the pre-registered balanced multinomial lbfgs probe and score validation."""
    ytr = np.asarray(y_train).astype(str)
    yva = np.asarray(y_validation).astype(str)
    if standardize_blocks:
        xtr, xva, scaler_stats = standardize_feature_blocks(train_blocks, validation_blocks)
    else:
        names = list(train_blocks)
        xtr = np.concatenate([np.asarray(train_blocks[n], dtype=np.float32) for n in names], axis=1)
        xva = np.concatenate([np.asarray(validation_blocks[n], dtype=np.float32) for n in names], axis=1)
        scaler_stats = None
    if xtr.shape[0] != ytr.size or xva.shape[0] != yva.size:
        raise ValueError("Probe feature and label row counts differ.")
    if not np.isfinite(xtr).all() or not np.isfinite(xva).all():
        raise FloatingPointError("Probe features contain NaN or Inf.")
    probe = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        probe.fit(xtr, ytr)
        prediction = probe.predict(xva).astype(str)
    fit_seconds = time.perf_counter() - start
    labels = np.unique(yva)
    precision, recall, f1, support = precision_recall_fscore_support(
        yva, prediction, labels=labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "prediction": prediction,
        "macro_f1": macro_f1_supported(yva, prediction, labels),
        "per_class": per_class,
        "validation_support": {str(label): int(np.sum(yva == label)) for label in labels},
        "fit_seconds": float(fit_seconds),
        "n_iter": probe.n_iter_.astype(int).tolist(),
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": None,
        "class_weight": "balanced",
        "standardized_blocks": bool(standardize_blocks),
        "scaler_stats": scaler_stats,
        "train_rows": int(ytr.size),
        "validation_rows": int(yva.size),
    }
