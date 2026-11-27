"""Optional label-free within-residual decorrelation objective for Stage 6."""
from __future__ import annotations

import torch


def residual_decorrelation_loss(
    residual: torch.Tensor, *, variance_floor: float
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Penalize off-diagonal residual-coordinate correlations per dataset.

    L_decorr = ||offdiag(C_R)||_F^2 / (m * (m - 1)).
    C_R is computed in float32 from this dataset's current sub-batch only.
    Dimensions below the same standard-deviation floor used by L_red are
    excluded. Call independently for each source dataset's 256-row batch.
    """
    if residual.ndim != 2:
        raise ValueError("R must be a rank-2 [batch, residual_dim] tensor.")
    if residual.shape[0] < 2:
        raise ValueError("Residual decorrelation requires batch size >= 2.")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive.")
    r = residual.to(dtype=torch.float32)
    centered = r - r.mean(dim=0, keepdim=True)
    covariance = (centered.transpose(0, 1) @ centered) / float(r.shape[0] - 1)
    variances = torch.diagonal(covariance)
    valid = variances >= float(variance_floor) ** 2
    active = int(valid.sum().item())
    masked = int((~valid).sum().item())
    if active < 2:
        zero = r.sum() * 0.0
        return zero, {
            "active_dimensions": active,
            "masked_dimensions": masked,
            "mean_abs_offdiag_correlation": 0.0,
        }
    active_centered = centered[:, valid]
    active_variances = variances[valid]
    safe_std = active_variances.clamp_min(float(variance_floor) ** 2).sqrt()
    standardized = active_centered / safe_std.unsqueeze(0)
    correlation = (standardized.transpose(0, 1) @ standardized) / float(r.shape[0] - 1)
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    denominator = float(active * (active - 1))
    loss = off_diagonal.square().sum() / denominator
    mean_abs = off_diagonal.detach().abs().sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite residual decorrelation loss.")
    return loss, {
        "active_dimensions": active,
        "masked_dimensions": masked,
        "mean_abs_offdiag_correlation": float(mean_abs.cpu()),
    }
