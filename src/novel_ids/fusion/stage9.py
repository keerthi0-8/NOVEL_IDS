"""Stage 9: 128-dimensional unified representation."""
from __future__ import annotations

import torch
from torch import nn


class UnifiedRepresentation(nn.Module):
    """Apply the approved scalar-compatibility fusion equation."""

    def forward(
        self,
        *,
        z_s: torch.Tensor,
        r: torch.Tensor,
        r_proj: torch.Tensor,
        gate: torch.Tensor,
        a: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
        z_f: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensors = {"Z_s": z_s, "R": r, "R_proj": r_proj, "G": gate, "A": a, "U": u}
        for name, value in tensors.items():
            if value.ndim != 2 or value.shape[1] != 128:
                raise ValueError(f"{name} must have shape [B,128].")
            if value.shape[0] != z_s.shape[0]:
                raise ValueError(f"{name} batch size must match Z_s.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite.")
        if c.shape != (z_s.shape[0], 1):
            raise ValueError("C must have scalar-per-sample shape [B,1].")
        c = c.detach().to(device=z_s.device, dtype=z_s.dtype)
        if not torch.isfinite(c).all() or torch.any((c < 0) | (c > 1)):
            raise ValueError("C must be finite and within [0,1].")
        if z_f.shape != z_s.shape or not torch.isfinite(z_f).all():
            raise ValueError("Z_f must be finite with shape [B,128].")

        z_u = z_s + gate * r_proj + c * a + (1.0 - c) * u
        return {
            "Z_u": z_u, "Z_s": z_s, "R": r, "R_proj": r_proj,
            "G": gate, "A": a, "U": u, "C": c, "Z_f": z_f,
        }
