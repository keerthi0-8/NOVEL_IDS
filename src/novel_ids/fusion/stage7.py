"""Stage 7: adaptive per-dimension residual fusion."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class AdaptiveResidualFusion(nn.Module):
    """Fuse a 128-D shared representation with a 128-D projected residual.

    The optional regularizer is supplied by the caller. No regularization
    formula or coefficient is selected here; the approved default is disabled.
    """

    def __init__(
        self,
        *,
        shared_dim: int = 128,
        residual_dim: int = 128,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if shared_dim != 128 or residual_dim != 128:
            raise ValueError("Stage 7 input and output dimensions must be 128.")
        self.shared_dim = shared_dim
        self.residual_dim = residual_dim
        self.gate = nn.Linear(shared_dim + residual_dim, residual_dim)
        nn.init.constant_(self.gate.bias, -2.0)
        self.regularizer = regularizer

    def forward(
        self, z_s: torch.Tensor, r_proj: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if z_s.ndim != 2 or r_proj.ndim != 2:
            raise ValueError("Z_s and R_proj must be rank-2 [batch, 128] tensors.")
        if z_s.shape != r_proj.shape or z_s.shape[1] != 128:
            raise ValueError("Z_s and R_proj must have matching [batch, 128] shapes.")
        gate = torch.sigmoid(self.gate(torch.cat((z_s, r_proj), dim=-1)))
        z_f = z_s + gate * r_proj
        return {"G": gate, "Z_f": z_f}

    def regularization_loss(self, gate: torch.Tensor) -> torch.Tensor | None:
        """Return caller-defined gate regularization, or None when disabled."""
        if self.regularizer is None:
            return None
        value = self.regularizer(gate)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("Gate regularizer must return a scalar tensor.")
        return value
