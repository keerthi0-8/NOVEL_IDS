"""Per-dataset classification head on Z_s (Stage 5A shared encoder output).

Follows NOVEL_IDS_MASTER_SPEC.md Section 6.1A: shared supervision with
class-weighted cross-entropy, classes derived dynamically from each dataset's
training labels.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Linear classification head on Z_s for one dataset's native labels."""

    def __init__(self, input_dim: int, num_classes: int, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes)."""
        return self.classifier(z_s)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Class-weighted cross-entropy loss.

        Weights are registered as a buffer and derived from training labels
        only; they are never updated by gradients.
        """
        if self.class_weights is not None:
            return nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
        return nn.functional.cross_entropy(logits, targets)


def compute_class_weights(
    labels: list[str | int],
    num_classes: int,
    clip_ratio: float = 20.0,
) -> torch.Tensor:
    """Compute inverse-square-root class weights from training labels only.

    Weights are normalized so the mean weight is 1.0 and clipped so the
    largest-to-smallest ratio does not exceed ``clip_ratio``.
    """
    import numpy as np

    label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
    counts = np.zeros(num_classes, dtype=np.float64)
    for label in labels:
        counts[label_to_index[label]] += 1.0

    # Avoid division by zero for classes with no training samples.
    counts = np.maximum(counts, 1.0)
    inv_sqrt_freq = 1.0 / np.sqrt(counts)
    weights = inv_sqrt_freq / inv_sqrt_freq.mean()

    # Clip ratio between largest and smallest.
    max_w = float(weights.max())
    min_w = float(weights.min())
    if min_w > 0 and max_w / min_w > clip_ratio:
        weights = weights * (clip_ratio * min_w / max_w)

    return torch.as_tensor(weights, dtype=torch.float32)
