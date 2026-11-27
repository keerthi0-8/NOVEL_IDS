"""Stage 5B reconstruction decoder D_5B (training only; removable at inference).

Follows NOVEL_IDS_MASTER_SPEC.md Section 6.1B: masked-feature modelling on
dataset-specific positions. Predicts feature f from Z_d and I_f (identity
vector). Numeric loss uses Huber on robust-scaled values; categorical loss uses
cross-entropy over the correct dataset vocabulary (UNKNOWN is not a target).

Only dataset-specific positions are reconstructed (not the five common
features). Unavailable or missing positions are excluded from the loss.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ReconstructionDecoder(nn.Module):
    """Identity-conditioned decoder predicting feature values from Z_d."""

    def __init__(
        self,
        z_dim: int,
        identity_dim: int,
        hidden_dim: int = 128,
        numeric_features: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.identity_dim = identity_dim
        self.hidden_dim = hidden_dim
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or {}

        self.trunk = nn.Sequential(
            nn.Linear(z_dim + identity_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.numeric_head = nn.Linear(hidden_dim, 1)
        self.categorical_heads = nn.ModuleDict()
        for position, vocab_size in self.categorical_features.items():
            self.categorical_heads[str(position)] = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z_d: torch.Tensor,
        identity_vectors: torch.Tensor,
        positions: list[int],
    ) -> dict[int, torch.Tensor]:
        """Decode feature values for the given positions.

        Args:
            z_d: Dataset-specific representation, shape (batch, z_dim).
            identity_vectors: All feature identities, shape (num_tokens,
                identity_dim). We index into this by position.
            positions: List of token positions to decode.

        Returns:
            Mapping from position to prediction tensor:
              - numeric: shape (batch, 1)
              - categorical: shape (batch, vocab_size) logits
        """
        predictions: dict[int, torch.Tensor] = {}
        for position in positions:
            identity = identity_vectors[position]  # (identity_dim,)
            identity_batch = identity.unsqueeze(0).expand(z_d.shape[0], -1)
            combined = torch.cat([z_d, identity_batch], dim=-1)
            trunk_output = self.trunk(combined)

            if position in self.numeric_features:
                predictions[position] = self.numeric_head(trunk_output)
            elif str(position) in self.categorical_heads:
                predictions[position] = self.categorical_heads[str(position)](trunk_output)
            else:
                raise ValueError(f"Position {position} is neither numeric nor categorical.")

        return predictions

    def reconstruction_loss(
        self,
        predictions: dict[int, torch.Tensor],
        targets: dict[int, torch.Tensor],
        masks: dict[int, torch.Tensor],
        numeric_weight: float = 1.0,
        categorical_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute reconstruction loss over valid masked positions."""
        numeric_losses: list[torch.Tensor] = []
        categorical_losses: list[torch.Tensor] = []

        for position, pred in predictions.items():
            mask = masks[position]
            if not mask.any():
                continue

            if position in self.numeric_features:
                target = targets[position].float()[mask].unsqueeze(-1)
                pred_valid = pred[mask]
                loss = nn.functional.huber_loss(pred_valid, target)
                numeric_losses.append(loss)
            else:
                target = targets[position].long()[mask]
                pred_valid = pred[mask]
                loss = nn.functional.cross_entropy(pred_valid, target)
                categorical_losses.append(loss)

        total_numeric = torch.stack(numeric_losses).mean() if numeric_losses else _zero()
        total_categorical = (
            torch.stack(categorical_losses).mean() if categorical_losses else _zero()
        )

        total = numeric_weight * total_numeric + categorical_weight * total_categorical

        logged: dict[str, float] = {
            "reconstruction_numeric": float(total_numeric.detach()),
            "reconstruction_categorical": float(total_categorical.detach()),
            "reconstruction_total": float(total.detach()),
        }
        return total, logged


def _zero() -> torch.Tensor:
    """Return a zero scalar tensor that requires grad."""
    return torch.tensor(0.0, requires_grad=True)