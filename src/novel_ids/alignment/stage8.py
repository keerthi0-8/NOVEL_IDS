"""Stage 8: compatibility-aware selective alignment (synthetic-testable core)."""
from __future__ import annotations

import math

import torch
from torch import nn


NSL_KDD = 0
UNSW_NB15 = 1
UNSEEN = 2
CORAL_LAMBDA_CANDIDATES = (0.0, 0.05, 0.2, 1.0)


def _require_matrix(name: str, value: torch.Tensor, width: int | None = None) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [batch, features].")
    if width is not None and value.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


class CompatibilityEstimator(nn.Module):
    """Frozen source-reference compatibility estimator.

    fit_source_training_references must receive source TRAINING Z_f tensors only.
    The class accepts no labels and never updates references during forward.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        alpha: float = 0.5,
        distance_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if latent_dim != 128:
            raise ValueError("Stage 8 latent dimension must be 128.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.latent_dim = latent_dim
        self.alpha = float(alpha)
        self.distance_epsilon = float(distance_epsilon)
        self.register_buffer("reference_means", torch.zeros(2, latent_dim))
        self.register_buffer("reference_variances", torch.ones(2, latent_dim))
        self.register_buffer("tau", torch.tensor(1.0))
        self.register_buffer("references_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_source_training_references(
        self, z_nsl_train: torch.Tensor, z_unsw_train: torch.Tensor
    ) -> None:
        """Fit frozen diagonal-Gaussian references from source training Z_f only."""
        _require_matrix("z_nsl_train", z_nsl_train, self.latent_dim)
        _require_matrix("z_unsw_train", z_unsw_train, self.latent_dim)
        if bool(self.references_fitted.item()):
            raise RuntimeError("Source training references are fit-once and already frozen.")
        if z_nsl_train.shape[0] < 2 or z_unsw_train.shape[0] < 2:
            raise ValueError("Each source training reference needs at least 2 rows.")
        nsl = z_nsl_train.detach().float()
        unsw = z_unsw_train.detach().float()
        means = torch.stack((nsl.mean(0), unsw.mean(0)))
        variances = torch.stack((nsl.var(0, unbiased=False), unsw.var(0, unbiased=False)))
        variances = variances.clamp_min(self.distance_epsilon)
        d_nsl_to_unsw = ((nsl - means[UNSW_NB15]).square() / variances[UNSW_NB15]).mean(1)
        d_unsw_to_nsl = ((unsw - means[NSL_KDD]).square() / variances[NSL_KDD]).mean(1)
        fitted_tau = torch.cat((d_nsl_to_unsw, d_unsw_to_nsl)).median()
        self.reference_means.copy_(means.to(self.reference_means.device))
        self.reference_variances.copy_(variances.to(self.reference_variances.device))
        self.tau.copy_(fitted_tau.clamp_min(self.distance_epsilon).to(self.tau.device))
        self.references_fitted.fill_(True)

    @staticmethod
    def confidence_score(
        available: torch.Tensor,
        match_level: torch.Tensor,
        match_confidence: torch.Tensor,
        group_only: torch.Tensor,
        *,
        common_feature_only: bool = False,
    ) -> torch.Tensor:
        """Return [B,1] average onboarding confidence over available features."""
        if available.ndim != 2 or match_level.shape != available.shape:
            raise ValueError("availability and match_level must have matching [B,F] shapes.")
        if match_confidence.shape != available.shape or group_only.shape != available.shape:
            raise ValueError("confidence and group_only must match [B,F].")
        if available.dtype != torch.bool or group_only.dtype != torch.bool:
            raise ValueError("available and group_only must be boolean tensors.")
        if not torch.isfinite(match_confidence).all():
            raise ValueError("match_confidence must be finite.")
        if torch.any((match_confidence < 0) | (match_confidence > 1)):
            raise ValueError("match_confidence must be in [0,1].")
        if torch.any((match_level < 1) | (match_level > 3)):
            raise ValueError("match_level values must be 1, 2, or 3.")

        level1 = match_level == 1
        level2_group = (match_level == 2) & group_only
        contributions = torch.where(
            level1,
            match_confidence,
            torch.where(level2_group, 0.5 * match_confidence, torch.zeros_like(match_confidence)),
        )
        if common_feature_only:
            contributions = torch.where(level1, match_confidence, torch.zeros_like(match_confidence))
        mask = available.to(contributions.dtype)
        if common_feature_only:
            mask = (available & level1).to(contributions.dtype)
        denominator = mask.sum(dim=1, keepdim=True)
        numerator = (contributions * mask).sum(dim=1, keepdim=True)
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.zeros_like(numerator))

    def _distance(self, z_f: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        if not bool(self.references_fitted.item()):
            raise RuntimeError("Source training references must be fitted before compatibility inference.")
        if source_ids.ndim != 1 or source_ids.shape[0] != z_f.shape[0]:
            raise ValueError("source_ids must have shape [B].")
        if torch.any((source_ids < NSL_KDD) | (source_ids > UNSEEN)):
            raise ValueError("source_ids must be 0 (NSL), 1 (UNSW), or 2 (unseen).")
        z = z_f.detach().float()
        means = self.reference_means.to(z.device)
        variances = self.reference_variances.to(z.device)
        d_to_nsl = ((z - means[NSL_KDD]).square() / variances[NSL_KDD]).mean(1, keepdim=True)
        d_to_unsw = ((z - means[UNSW_NB15]).square() / variances[UNSW_NB15]).mean(1, keepdim=True)
        known_nsl = source_ids == NSL_KDD
        known_unsw = source_ids == UNSW_NB15
        unseen = source_ids == UNSEEN
        return torch.where(
            known_nsl[:, None],
            d_to_unsw,
            torch.where(known_unsw[:, None], d_to_nsl, torch.minimum(d_to_nsl, d_to_unsw)),
        )

    def forward(
        self,
        z_f: torch.Tensor,
        source_ids: torch.Tensor,
        available: torch.Tensor,
        match_level: torch.Tensor,
        match_confidence: torch.Tensor,
        group_only: torch.Tensor,
        *,
        common_feature_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        _require_matrix("z_f", z_f, self.latent_dim)
        with torch.no_grad():
            s_conf = self.confidence_score(
                available, match_level, match_confidence, group_only,
                common_feature_only=common_feature_only,
            )
            d = self._distance(z_f, source_ids.to(z_f.device))
            s_dist = torch.exp(-d / self.tau.to(z_f.device).clamp_min(self.distance_epsilon))
            c = (s_conf.pow(self.alpha) * s_dist.pow(1.0 - self.alpha)).clamp(0.05, 0.95)
        return {
            "C": c.detach(),
            "s_conf": s_conf.detach(),
            "s_dist": s_dist.detach(),
            "distance": d.detach(),
            "tau": self.tau.detach().clone(),
        }


def weighted_covariance(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Centered weighted population covariance; constant weights equal unweighted covariance."""
    _require_matrix("values", values)
    if weights.ndim != 2 or weights.shape != (values.shape[0], 1):
        raise ValueError("weights must have shape [B,1].")
    x = values.float()
    w = weights.detach().float().clamp_min(0.0)
    total = w.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        raise ValueError("weighted covariance requires positive finite total weight.")
    mean = (x * w).sum(0, keepdim=True) / total
    centered = x - mean
    return (centered * w).T @ centered / total


def normalized_weighted_coral(
    a_nsl: torch.Tensor,
    a_unsw: torch.Tensor,
    c_nsl: torch.Tensor,
    c_unsw: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalized CORAL discrepancy between two source datasets."""
    _require_matrix("a_nsl", a_nsl, 128)
    _require_matrix("a_unsw", a_unsw, 128)
    if a_nsl.shape[0] < 256 or a_unsw.shape[0] < 256:
        raise ValueError("CORAL requires at least 256 samples per source sub-batch.")
    s_nsl = weighted_covariance(a_nsl, c_nsl)
    s_unsw = weighted_covariance(a_unsw, c_unsw)
    numerator = (s_nsl - s_unsw).square().sum()
    denominator = s_nsl.square().sum() + s_unsw.square().sum() + eps
    return numerator / denominator


class Stage8SelectiveAlignment(nn.Module):
    """Stage 8 projections, compatibility, CORAL objective, and ablation switches."""

    def __init__(
        self,
        *,
        coral_lambda: float = 0.0,
        mode: str = "compatibility",
        warmup_fraction: float = 1.0,
        per_dimension_compatibility: bool = False,
    ) -> None:
        super().__init__()
        if coral_lambda not in CORAL_LAMBDA_CANDIDATES:
            raise ValueError(f"coral_lambda must be one of {CORAL_LAMBDA_CANDIDATES}.")
        if mode not in {"compatibility", "no_alignment", "common_feature_only", "full_alignment"}:
            raise ValueError("Unsupported Stage 8 mode.")
        if not 0.0 <= warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be in [0,1].")
        self.per_dimension_compatibility = bool(per_dimension_compatibility)
        if self.per_dimension_compatibility:
            raise NotImplementedError(
                "Per-dimension [B,128] compatibility is disabled pending a separately approved formula."
            )
        self.coral_lambda = float(coral_lambda)
        self.mode = mode
        self.warmup_fraction = float(warmup_fraction)
        self.compatibility = CompatibilityEstimator()
        self.W_a = nn.Linear(128, 128)
        self.W_u = nn.Linear(128, 128)

    @staticmethod
    def _effective_c(raw_c: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "no_alignment":
            return torch.zeros_like(raw_c)
        if mode == "full_alignment":
            return torch.ones_like(raw_c)
        return raw_c.detach()

    def forward(
        self,
        z_f: torch.Tensor,
        z_d: torch.Tensor,
        source_ids: torch.Tensor,
        available: torch.Tensor,
        match_level: torch.Tensor,
        match_confidence: torch.Tensor,
        group_only: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require_matrix("z_f", z_f, 128)
        _require_matrix("z_d", z_d, 128)
        if z_f.shape != z_d.shape:
            raise ValueError("z_f and z_d must have matching [B,128] shapes.")
        raw = self.compatibility(
            z_f, source_ids, available, match_level, match_confidence, group_only,
            common_feature_only=(self.mode == "common_feature_only"),
        )
        c = self._effective_c(raw["C"], self.mode)
        a = self.W_a(z_f)
        u = self.W_u(z_d)
        return {
            "A": a, "U": u, "C": c.detach(), "C_raw": raw["C"],
            "s_conf": raw["s_conf"], "s_dist": raw["s_dist"],
            "distance": raw["distance"], "tau": raw["tau"],
        }

    def coral_loss(
        self,
        output: dict[str, torch.Tensor],
        source_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a = output["A"]
        c = output["C"]
        source_ids = source_ids.to(a.device)
        nsl = source_ids == NSL_KDD
        unsw = source_ids == UNSW_NB15
        zero = a.sum() * 0.0
        if self.mode == "no_alignment" or self.coral_lambda == 0.0:
            return {"L_coral": zero, "L_align": zero}
        if not nsl.any() or not unsw.any():
            raise ValueError("CORAL requires both source datasets in the batch.")
        weights_nsl = c[nsl]
        weights_unsw = c[unsw]
        if self.mode == "full_alignment":
            weights_nsl = torch.ones_like(weights_nsl)
            weights_unsw = torch.ones_like(weights_unsw)
        l_coral = normalized_weighted_coral(
            a[nsl], a[unsw], weights_nsl, weights_unsw
        )
        return {
            "L_coral": l_coral,
            "L_align": l_coral * self.coral_lambda * self.warmup_fraction,
        }
