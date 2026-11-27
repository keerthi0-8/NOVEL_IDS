"""Stage 10: feature-token conditioning and masked Transformer encoder.

Inputs are produced by the existing Stage 5 dual encoder and Stage 9 fusion.
This module does not own, import, or modify Stage 5 parameters.
"""
from __future__ import annotations

import torch
from torch import nn


class Stage10Transformer(nn.Module):
    """Variable-length feature-token Transformer with a learned CLS projection."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 128
        self.token_conditioning = nn.Linear(128, 128)
        self.cls_projection = nn.Linear(128, 128)

        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=2,
            norm=None,
            enable_nested_tensor=False,
        )

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters owned by this Stage 10 module."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _validate_inputs(
        tokens: torch.Tensor,
        z_u: torch.Tensor,
        m_avail: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be a tensor with shape [B,F,128].")
        batch, feature_count, hidden = tokens.shape
        if feature_count < 1 or hidden != 128:
            raise ValueError("tokens must have at least one feature and hidden dimension 128.")
        if z_u.shape != (batch, 128):
            raise ValueError(f"Z_u must have shape [{batch},128].")
        if m_avail.shape != (batch, feature_count):
            raise ValueError(f"M_avail must have shape [{batch},{feature_count}].")
        if tokens.device != z_u.device or tokens.device != m_avail.device:
            raise ValueError("tokens, Z_u, and M_avail must be on the same device.")
        if not torch.isfinite(tokens).all() or not torch.isfinite(z_u).all():
            raise ValueError("tokens and Z_u must contain only finite values.")
        if m_avail.dtype == torch.bool:
            available = m_avail
        else:
            if not torch.isfinite(m_avail).all():
                raise ValueError("M_avail must be finite.")
            if torch.any((m_avail != 0) & (m_avail != 1)):
                raise ValueError("M_avail must be a binary availability mask.")
            available = m_avail.to(dtype=torch.bool)
        return available

    def forward(
        self,
        tokens: torch.Tensor,
        z_u: torch.Tensor,
        m_avail: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return CLS representation Z_t and masked feature-token outputs."""
        available = self._validate_inputs(tokens, z_u, m_avail)
        conditioned = tokens + self.token_conditioning(z_u).unsqueeze(1)
        cls = self.cls_projection(z_u).unsqueeze(1)
        sequence = torch.cat((cls, conditioned), dim=1)

        # True in PyTorch key_padding_mask means the key is ignored.
        # CLS is prepended and always unmasked, including all-unavailable rows.
        cls_unmasked = torch.zeros(
            (available.shape[0], 1), dtype=torch.bool, device=available.device
        )
        key_padding_mask = torch.cat((cls_unmasked, ~available), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)

        z_t = encoded[:, 0, :]
        token_outputs = encoded[:, 1:, :].masked_fill(~available.unsqueeze(-1), 0.0)
        return {
            "Z_t": z_t,
            "token_outputs": token_outputs,
        }
