"""Stage 6 residual extraction primitives.

Stage 6 residual-learning primitives. Stage 5 remains frozen; Stage 6
training orchestration is implemented separately in stage6_training.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def freeze_stage5(stage5: nn.Module) -> nn.Module:
    """Freeze Stage 5 parameters and force evaluation mode."""
    stage5.eval()
    for parameter in stage5.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return stage5


def stage5_representations(stage5: nn.Module, *args, **kwargs):
    """Run a frozen Stage 5 module without constructing an autograd graph."""
    if stage5.training:
        raise RuntimeError("Stage 5 must be in eval mode before inference.")
    if any(parameter.requires_grad for parameter in stage5.parameters()):
        raise RuntimeError("All Stage 5 parameters must have requires_grad=False.")
    with torch.no_grad():
        return stage5(*args, **kwargs)


class ResidualPredictor(nn.Module):
    """Predict Z_d from detached Z_s; only this module is trained by L_pred."""

    def __init__(self, shared_dim: int, hidden_dim: int, dataset_dim: int):
        super().__init__()
        if min(shared_dim, hidden_dim, dataset_dim) <= 0:
            raise ValueError("All dimensions must be positive.")
        self.network = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dataset_dim),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.network(z_s.detach())


class ResidualEncoder(nn.Module):
    """Bottleneck encoder h(E_d) producing R."""

    def __init__(self, input_dim: int, hidden_dim: int, residual_dim: int):
        super().__init__()
        if min(input_dim, hidden_dim, residual_dim) <= 0:
            raise ValueError("All dimensions must be positive.")
        self.input_dim = input_dim
        self.residual_dim = residual_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, residual_dim),
        )

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        return self.network(error)


class ResidualDecoder(nn.Module):
    """Training-only decoder dec_R(R) -> E_d_hat."""

    def __init__(self, residual_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if min(residual_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("All dimensions must be positive.")
        self.network = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.network(residual)


def initialize_fixed_semi_orthogonal_projection(
    projection: nn.Module,
) -> nn.Module:
    """Initialize and freeze the 64->128 fixed semi-orthogonal baseline.

    The linear weight has shape [128, 64]. Orthogonal initialization makes
    its columns orthonormal (W^T W = I_64); bias is zero. Identity projections
    for residual_dim=128 need no initialization.
    """
    if isinstance(projection, nn.Identity):
        return projection
    if not isinstance(projection, nn.Linear):
        raise TypeError("Projection must be nn.Linear(64, 128) or nn.Identity.")
    if projection.in_features != 64 or projection.out_features != 128:
        raise ValueError("Fixed semi-orthogonal baseline requires Linear(64, 128).")
    with torch.no_grad():
        nn.init.orthogonal_(projection.weight)
        if projection.bias is not None:
            projection.bias.zero_()
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return projection


def projection_geometry_loss(
    residual: torch.Tensor,
    projected_residual: torch.Tensor,
) -> torch.Tensor:
    """Approved label-free Gram-preservation objective.

    L_proj = ||K_P - StopGrad(K_R)||_F^2 / B^2,
    K_R = R_c R_c^T / (B-1), K_P = P_c P_c^T / (B-1).

    Both the source representation and target Gram matrix are detached, so
    gradients flow only through projected_residual (and hence the projection).
    Call independently for each dataset batch.
    """
    if residual.ndim != 2 or projected_residual.ndim != 2:
        raise ValueError("R and R_proj must be rank-2 [batch, feature] tensors.")
    if residual.shape[0] != projected_residual.shape[0]:
        raise ValueError("R and R_proj batch sizes must match.")
    batch = residual.shape[0]
    if batch < 2:
        raise ValueError("Projection geometry loss requires batch size >= 2.")
    r = residual.detach().to(dtype=torch.float32)
    p = projected_residual.to(dtype=torch.float32)
    r_centered = r - r.mean(dim=0, keepdim=True)
    p_centered = p - p.mean(dim=0, keepdim=True)
    k_r = (r_centered @ r_centered.transpose(0, 1)) / float(batch - 1)
    k_p = (p_centered @ p_centered.transpose(0, 1)) / float(batch - 1)
    return (k_p - k_r.detach()).square().sum() / float(batch * batch)


class ResidualProjection(nn.Module):
    """Project selected Stage 6 residual width to the approved 128-D interface."""

    def __init__(self, residual_dim: int, output_dim: int = 128):
        super().__init__()
        if residual_dim not in {64, 128}:
            raise ValueError("Initial residual width must be 64 or 128.")
        if output_dim != 128:
            raise ValueError("Stage 7/9 residual interface is fixed at 128 dimensions.")
        self.projection = (
            nn.Identity() if residual_dim == output_dim
            else nn.Linear(residual_dim, output_dim)
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.projection(residual)


class Stage6ResidualBranch(nn.Module):
    """Compose predictor, residual encoder, and removable reconstruction decoder."""

    def __init__(
        self,
        shared_dim: int,
        dataset_dim: int,
        predictor_hidden_dim: int,
        residual_hidden_dim: int,
        residual_dim: int,
        decoder_hidden_dim: int,
    ):
        super().__init__()
        if residual_dim not in {64, 128}:
            raise ValueError("residual_dim must be 64 or 128.")
        self.predictor = ResidualPredictor(
            shared_dim, predictor_hidden_dim, dataset_dim
        )
        self.encoder = ResidualEncoder(
            dataset_dim, residual_hidden_dim, residual_dim
        )
        self.decoder = ResidualDecoder(
            residual_dim, decoder_hidden_dim, dataset_dim
        )
        self.projection = ResidualProjection(residual_dim, 128)

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_s.ndim != 2 or z_d.ndim != 2:
            raise ValueError("Z_s and Z_d must be rank-2 [batch, feature] tensors.")
        if z_s.shape[0] != z_d.shape[0]:
            raise ValueError("Z_s and Z_d batch sizes must match.")
        z_d_hat = self.predictor(z_s.detach())
        # Detach predictor output so residual objectives cannot train g.
        e_d = z_d.detach() - z_d_hat.detach()
        residual = self.encoder(e_d)
        e_d_hat = self.decoder(residual)
        return {
            "Z_d_hat": z_d_hat,
            "E_d": e_d,
            "R": residual,
            # Normal inference path remains differentiable for later approved stages.
            "R_proj": self.projection(residual),
            # Projection-objective path detaches R so L_proj updates W_p only.
            "R_proj_objective": self.projection(residual.detach()),
            "E_d_hat": e_d_hat,
        }


def prediction_loss(z_d_hat: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
    """L_pred=MSE(Zhat_d, StopGrad(Z_d)); Z_s is detached in the predictor."""
    return F.mse_loss(z_d_hat, z_d.detach())


def information_reconstruction_loss(
    e_d_hat: torch.Tensor, e_d: torch.Tensor
) -> torch.Tensor:
    """Approved L_info=MSE(E_d_hat, StopGrad(E_d))."""
    return F.mse_loss(e_d_hat, e_d.detach())


def magnitude_stability_loss(
    residual: torch.Tensor,
    *,
    min_dimension_std: float = 0.1,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Approved Stage 6 magnitude objective, evaluated per dataset.

    L_mag = (RMS(R) - 1)^2
            + mean_j ReLU(min_dimension_std - std_batch(R[:, j]))^2

    Use population standard deviation (correction=0) so small synthetic
    batches remain finite. Call separately for each source dataset.
    """
    if residual.ndim != 2:
        raise ValueError("R must be a rank-2 [batch, residual_dim] tensor.")
    if residual.shape[0] < 2:
        raise ValueError("Magnitude stability requires at least two rows.")
    if min_dimension_std < 0 or epsilon <= 0:
        raise ValueError("min_dimension_std must be non-negative and epsilon positive.")
    r = residual.float()
    rms = torch.sqrt(r.square().mean() + epsilon)
    dimension_std = torch.sqrt(r.var(dim=0, correction=0) + epsilon)
    magnitude_term = (rms - 1.0).square()
    variance_term = torch.relu(min_dimension_std - dimension_std).square().mean()
    loss = magnitude_term + variance_term
    stats = {
        "rms": float(rms.detach().cpu()),
        "dimension_std_min": float(dimension_std.min().detach().cpu()),
        "dimension_std_mean": float(dimension_std.mean().detach().cpu()),
        "magnitude_term": float(magnitude_term.detach().cpu()),
        "variance_term": float(variance_term.detach().cpu()),
    }
    return loss, stats


def standardized_cross_covariance_loss(
    z_s: torch.Tensor,
    residual: torch.Tensor,
    *,
    variance_floor: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Per-dataset float32 standardized cross-covariance penalty.

    Call separately for each dataset sub-batch; never concatenate datasets.
    The caller is responsible for supplying the approved 256-row sub-batch.
    Dimensions below variance_floor are masked from the covariance.
    """
    if z_s.ndim != 2 or residual.ndim != 2:
        raise ValueError("Z_s and R must be rank-2 tensors.")
    if z_s.shape[0] != residual.shape[0]:
        raise ValueError("Z_s and R batch sizes must match.")
    batch = z_s.shape[0]
    if batch < 2:
        raise ValueError("Cross-covariance requires batch size >= 2.")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive.")

    zs = z_s.detach().to(dtype=torch.float32)
    r = residual.to(dtype=torch.float32)
    zs_centered = zs - zs.mean(dim=0, keepdim=True)
    r_centered = r - r.mean(dim=0, keepdim=True)
    # Clamp variance before sqrt so constant dimensions have finite backward
    # derivatives; validity retains the original standard-deviation threshold.
    variance_floor_sq = variance_floor ** 2
    zs_var = zs_centered.square().mean(dim=0)
    r_var = r_centered.square().mean(dim=0)
    zs_valid = zs_var >= variance_floor_sq
    r_valid = r_var >= variance_floor_sq
    zs_std = zs_var.clamp_min(variance_floor_sq).sqrt()
    r_std = r_var.clamp_min(variance_floor_sq).sqrt()
    zs_tilde = torch.where(
        zs_valid.unsqueeze(0),
        zs_centered / zs_std.clamp_min(variance_floor).unsqueeze(0),
        torch.zeros_like(zs_centered),
    )
    r_tilde = torch.where(
        r_valid.unsqueeze(0),
        r_centered / r_std.clamp_min(variance_floor).unsqueeze(0),
        torch.zeros_like(r_centered),
    )
    covariance = (zs_tilde.transpose(0, 1) @ r_tilde) / float(batch - 1)
    loss = covariance.square().sum() / float(zs.shape[1] * r.shape[1])
    stats = {
        "masked_shared_dimensions": int((~zs_valid).sum().item()),
        "masked_residual_dimensions": int((~r_valid).sum().item()),
    }
    return loss, stats


def redundancy_warmup(
    step: int, warmup_steps: int, target_coefficient: float
) -> float:
    """Linear coefficient warm-up; values are supplied by experiment config."""
    if step < 0 or warmup_steps < 0 or target_coefficient < 0:
        raise ValueError("Warm-up arguments must be non-negative.")
    if warmup_steps == 0:
        return float(target_coefficient)
    return float(target_coefficient) * min(step / warmup_steps, 1.0)


@dataclass(frozen=True)
class Stage6LossCoefficients:
    """Explicit experiment-config coefficients; no research defaults are implied."""

    prediction: float
    information: float
    magnitude: float
    redundancy: float

    def __post_init__(self):
        if min(self.prediction, self.information, self.magnitude, self.redundancy) < 0:
            raise ValueError("Stage 6 loss coefficients must be non-negative.")


def combine_stage6_losses_with_projection(
    *,
    l_pred: torch.Tensor,
    l_info: torch.Tensor,
    l_mag: torch.Tensor,
    l_red: torch.Tensor,
    l_proj: torch.Tensor,
    coefficients: Stage6LossCoefficients,
    projection_coefficient: float = 1.0,
) -> torch.Tensor:
    """Add the separately approved projection objective without altering the four losses."""
    if projection_coefficient < 0:
        raise ValueError("projection_coefficient must be non-negative.")
    base = combine_stage6_losses(
        l_pred=l_pred,
        l_info=l_info,
        l_mag=l_mag,
        l_red=l_red,
        coefficients=coefficients,
    )
    return base + projection_coefficient * l_proj


def combine_stage6_losses(
    *,
    l_pred: torch.Tensor,
    l_info: torch.Tensor,
    l_mag: torch.Tensor,
    l_red: torch.Tensor,
    coefficients: Stage6LossCoefficients,
) -> torch.Tensor:
    """Combine the four approved Stage 6 losses with explicit coefficients."""
    return (
        coefficients.prediction * l_pred
        + coefficients.information * l_info
        + coefficients.magnitude * l_mag
        + coefficients.redundancy * l_red
    )
