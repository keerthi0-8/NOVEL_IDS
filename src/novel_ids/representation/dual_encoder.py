from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from novel_ids.representation.feature_identity import FeatureIdentityEncoder


TOKEN_COUNT = 78
IDENTITY_DIM = 128
OUTPUT_DIM = 128

# Positions follow the saved unified_feature_schema.yaml.
CATEGORICAL_CONFIG = {
    "nsl_kdd": {"positions": [1, 2, 5], "vocab_sizes": [3, 70, 11]},
    "unsw_nb15": {"positions": [1, 2, 41], "vocab_sizes": [133, 13, 8]},
}

REVIEWED_GROUPS = (
    "connection_state",
    "host_and_privilege_activity",
    "authentication_activity",
    "connection_history_count",
    "connection_error_rate",
    "connection_context_ratio",
    "packet_volume",
    "ttl_measurement",
    "bandwidth_load",
    "packet_loss_or_retransmission",
    "packet_interarrival_time",
    "packet_jitter",
    "tcp_window_measurement",
    "tcp_handshake_timing",
    "packet_size_measurement",
    "application_protocol_activity",
    "application_content_size",
    "state_ttl_condition_count",
)


class SharedContextAdapter(nn.Module):
    """Lightweight shared 5B adapter conditioned by onboarding context."""

    def __init__(self, context_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(OUTPUT_DIM, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.norm = nn.LayerNorm(OUTPUT_DIM)

    def forward(self, shared: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(shared)
        gamma, beta = self.context_projection(context).chunk(2, dim=-1)
        hidden = torch.nn.functional.gelu(hidden * (1.0 + gamma) + beta)
        return self.norm(self.output_projection(self.dropout(hidden)))


class DualEncoder(nn.Module):
    """Variable-feature Stage 5A shared encoder and shared 5B adapter.

    The existing 78-token input remains valid. Extra tokens are appended and
    have source index -1. Their identities come from the same descriptor
    encoder as source identities and never receive a source-specific offset.
    """

    def __init__(
        self,
        feature_identity_vectors: Any,
        dropout: float = 0.1,
        feature_identity_encoder: FeatureIdentityEncoder | None = None,
        source_descriptors: list[dict[str, Any]] | None = None,
        feature_set_dropout_rate: float = 0.1,
        feature_set_dropout_all_probability: float = 0.05,
        value_masking_enabled: bool = True,
        value_masking_rate: float = 0.05,
        augmentation_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        if augmentation_config is not None:
            value_config = augmentation_config.get("value_masking", {})
            feature_config = augmentation_config.get("feature_set_dropout", {})
            value_masking_enabled = bool(value_config.get("enabled", value_masking_enabled))
            value_masking_rate = float(value_config.get("probability", value_masking_rate))
            feature_set_dropout_rate = (
                float(feature_config.get("probability", feature_set_dropout_rate))
                if feature_config.get("enabled", True) else 0.0
            )
            feature_set_dropout_all_probability = float(
                feature_config.get(
                    "all_dataset_specific_probability",
                    feature_set_dropout_all_probability,
                )
            )
        identities = torch.as_tensor(feature_identity_vectors, dtype=torch.float32)
        if identities.shape != (TOKEN_COUNT, IDENTITY_DIM):
            raise ValueError(
                f"Expected feature identity matrix with shape ({TOKEN_COUNT}, {IDENTITY_DIM}); "
                f"received {tuple(identities.shape)}."
            )
        if not torch.isfinite(identities).all():
            raise ValueError("Feature identity matrix contains NaN or Inf.")
        if not 0.05 <= value_masking_rate <= 0.10:
            raise ValueError("value_masking_rate must be between 0.05 and 0.10.")
        if not 0.0 <= feature_set_dropout_rate <= 1.0:
            raise ValueError("feature_set_dropout_rate must be in [0, 1].")
        if not 0.0 <= feature_set_dropout_all_probability <= 1.0:
            raise ValueError("feature_set_dropout_all_probability must be in [0, 1].")

        self.feature_identity_encoder = feature_identity_encoder
        self.source_descriptors = source_descriptors
        if source_descriptors is not None and len(source_descriptors) != TOKEN_COUNT:
            raise ValueError("source_descriptors must contain exactly 78 entries.")

        if feature_identity_encoder is not None and source_descriptors is not None:
            with torch.no_grad():
                descriptor_base = feature_identity_encoder.encode(source_descriptors).detach()
            if descriptor_base.shape != identities.shape:
                raise ValueError("Source descriptor identities must have shape (78, 128).")
            initial_offsets = identities - descriptor_base
            # Level 1 common features have one shared identity across datasets.
            for index, descriptor in enumerate(source_descriptors):
                if descriptor.get("level") == 1 and descriptor.get("semantic_id"):
                    initial_offsets[index].zero_()
            base = descriptor_base
        else:
            base = identities
            initial_offsets = torch.zeros_like(identities)

        self.register_buffer("source_identity_base", base.clone())
        self.source_identity_offsets = nn.Parameter(initial_offsets.clone())

        groups = set(REVIEWED_GROUPS)
        if source_descriptors:
            groups.update(
                str(d["semantic_group"])
                for d in source_descriptors
                if d.get("level") == 2 and d.get("semantic_group")
            )
        self.group_names = tuple(sorted(groups))
        self.group_to_index = {name: i + 1 for i, name in enumerate(self.group_names)}
        self.group_embeddings = nn.Embedding(len(self.group_names) + 1, IDENTITY_DIM, padding_idx=0)
        nn.init.normal_(self.group_embeddings.weight, std=0.02)
        with torch.no_grad():
            self.group_embeddings.weight[0].zero_()

        self.numeric_projection = nn.Sequential(
            nn.Linear(IDENTITY_DIM + 1, IDENTITY_DIM),
            nn.GELU(),
            nn.Linear(IDENTITY_DIM, IDENTITY_DIM),
        )
        self.categorical_embeddings = nn.ModuleDict()
        for dataset, config in CATEGORICAL_CONFIG.items():
            self.categorical_embeddings[dataset] = nn.ModuleList(
                [nn.Embedding(size + 1, IDENTITY_DIM) for size in config["vocab_sizes"]]
            )
        self.unknown_category_embedding = nn.Parameter(torch.empty(IDENTITY_DIM))
        self.unavailable_embedding = nn.Parameter(torch.empty(IDENTITY_DIM))
        self.missing_value_embedding = nn.Parameter(torch.empty(IDENTITY_DIM))
        nn.init.normal_(self.unknown_category_embedding, std=0.02)
        nn.init.normal_(self.unavailable_embedding, std=0.02)
        nn.init.normal_(self.missing_value_embedding, std=0.02)

        self.mask_projection = nn.Linear(2, IDENTITY_DIM)
        self.shared_encoder = nn.Sequential(
            nn.Linear(IDENTITY_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, OUTPUT_DIM),
            nn.LayerNorm(OUTPUT_DIM),
        )
        context_dim = IDENTITY_DIM + len(self.group_names) + 1
        self.shared_context_adapter = SharedContextAdapter(context_dim, dropout=dropout)

        self.feature_set_dropout_rate = feature_set_dropout_rate
        self.feature_set_dropout_all_probability = feature_set_dropout_all_probability
        self.value_masking_enabled = value_masking_enabled
        self.value_masking_rate = value_masking_rate

    @property
    def feature_identities(self) -> torch.Tensor:
        """Backward-compatible view of the initialized source identities."""
        return self.source_identity_base + self.source_identity_offsets

    @staticmethod
    def _tensor(value: Any, device: torch.device, dtype=None) -> torch.Tensor:
        result = torch.as_tensor(value, device=device)
        return result.to(dtype=dtype) if dtype is not None else result

    def _source_indices(self, count: int, supplied: Any, device: torch.device) -> torch.Tensor:
        if supplied is None:
            if count > TOKEN_COUNT:
                return torch.cat(
                    [torch.arange(TOKEN_COUNT, device=device),
                     torch.full((count - TOKEN_COUNT,), -1, device=device)]
                ).long()
            return torch.arange(count, device=device)
        indices = self._tensor(supplied, device, torch.long)
        if indices.shape != (count,) or torch.any(indices < -1) or torch.any(indices >= TOKEN_COUNT):
            raise ValueError("source_feature_indices must have shape (tokens,) and values -1 or 0..77.")
        return indices

    def _identity_tokens(self, count, indices, descriptors, device):
        if descriptors is None:
            if count > TOKEN_COUNT or torch.any(indices == -1):
                raise ValueError("Descriptors are required for extension feature tokens.")
            base = self.source_identity_base[indices]
        else:
            if len(descriptors) != count:
                raise ValueError("feature_descriptors must contain one descriptor per token.")
            if self.feature_identity_encoder is None:
                raise ValueError("A fitted FeatureIdentityEncoder is required for descriptor tokens.")
            base = self.feature_identity_encoder.encode(descriptors)
            if base.shape != (count, IDENTITY_DIM):
                raise ValueError("Descriptor encoder returned an unexpected identity shape.")
        source_specific = torch.tensor(
            [d.get("level") != 1 or not d.get("semantic_id") for d in descriptors]
            if descriptors is not None else [True] * count,
            device=device,
            dtype=torch.bool,
        )
        apply_offset = (indices >= 0) & source_specific
        offsets = torch.zeros_like(base)
        if apply_offset.any():
            offsets[apply_offset] = self.source_identity_offsets[indices[apply_offset]]
        return base + offsets, base

    def _group_ids(self, count, descriptors, device):
        ids = torch.zeros(count, dtype=torch.long, device=device)
        if descriptors is None:
            return ids
        for position, descriptor in enumerate(descriptors):
            if (
                descriptor.get("level") == 2
                and descriptor.get("review_status") == "approved"
                and descriptor.get("relationship") in {"group_only", "partial"}
            ):
                group = descriptor.get("semantic_group")
                if group and group in self.group_to_index:
                    ids[position] = self.group_to_index[group]
        return ids

    def _context(self, descriptor_base, group_ids, descriptors, available):
        batch, count = available.shape
        available_bool = available.bool()
        weights = available_bool.to(descriptor_base.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_descriptor = torch.einsum("bn,nd->bd", weights, descriptor_base) / denominator

        group_presence = []
        for group_id in range(1, len(self.group_names) + 1):
            belongs = (group_ids == group_id).unsqueeze(0)
            group_presence.append((available_bool & belongs).sum(dim=1, keepdim=True).to(descriptor_base.dtype) / denominator)
        group_fractions = (
            torch.cat(group_presence, dim=1)
            if group_presence else descriptor_base.new_zeros((batch, 0))
        )

        skew_values = descriptor_base.new_zeros(count)
        if descriptors is not None:
            values = []
            for descriptor in descriptors:
                value = (descriptor.get("statistics") or {}).get("skewness", 0.0)
                try:
                    value = float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    value = 0.0
                values.append(value if np.isfinite(value) else 0.0)
            skew_values = torch.tensor(values, device=descriptor_base.device, dtype=descriptor_base.dtype)
        mean_skew = (weights * skew_values.unsqueeze(0)).sum(dim=1, keepdim=True) / denominator
        return torch.cat([pooled_descriptor, group_fractions, mean_skew], dim=1)

    def forward(
        self,
        numeric_values: Any,
        categorical_values: Any,
        M_avail: Any,
        M_miss: Any,
        numeric_token_mask: Any,
        categorical_token_mask: Any,
        dataset: str | None = None,
        feature_descriptors: list[dict[str, Any]] | None = None,
        source_feature_indices: Any = None,
        is_unseen: bool = False,
    ) -> dict[str, torch.Tensor]:
        if is_unseen and (self.training or torch.is_grad_enabled()):
            raise ValueError("Unseen inference requires eval mode and torch.no_grad(); unseen data cannot update parameters.")
        if is_unseen and (feature_descriptors is None or source_feature_indices is None):
            raise ValueError("Unseen inference requires onboarding descriptors and explicit source indices.")

        device = self.source_identity_base.device
        numeric = self._tensor(numeric_values, device, torch.float32)
        categorical = self._tensor(categorical_values, device, torch.long)
        available = self._tensor(M_avail, device, torch.float32).clone()
        missing = self._tensor(M_miss, device, torch.float32).clone()
        numeric_mask = self._tensor(numeric_token_mask, device, torch.bool)
        categorical_mask = self._tensor(categorical_token_mask, device, torch.bool)

        if numeric.ndim != 2 or categorical.ndim != 2:
            raise ValueError("numeric_values and categorical_values must be 2D arrays.")
        if numeric.shape != categorical.shape:
            raise ValueError("Numeric and categorical arrays must have matching shapes.")
        batch_size, count = numeric.shape
        expected = (batch_size, count)
        for name, value in {"M_avail": available, "M_miss": missing}.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}; received {tuple(value.shape)}.")
        if numeric_mask.shape != (count,) or categorical_mask.shape != (count,):
            raise ValueError(f"Token type masks must have shape ({count},).")
        if torch.any(numeric_mask & categorical_mask):
            raise ValueError("Numeric and categorical token masks overlap.")
        if not torch.isfinite(available).all() or not torch.isfinite(missing).all():
            raise ValueError("Availability and missingness masks must be finite.")
        if torch.any((available < 0) | (available > 1)) or torch.any((missing < 0) | (missing > 1)):
            raise ValueError("Availability and missingness masks must be in [0, 1].")
        if torch.any(missing > available):
            raise ValueError("M_miss cannot exceed M_avail.")
        if not torch.all((available == 0) | (available == 1)) or not torch.all((missing == 0) | (missing == 1)):
            raise ValueError("M_avail and M_miss must be binary.")
        if torch.any((available > 0) & ~(numeric_mask | categorical_mask).unsqueeze(0)):
            raise ValueError("An available value must have a numeric or categorical token type.")

        if feature_descriptors is None and count == TOKEN_COUNT and self.source_descriptors is not None:
            feature_descriptors = self.source_descriptors
        indices = self._source_indices(count, source_feature_indices, device)
        if is_unseen and torch.any(indices >= 0):
            raise ValueError("Unseen features cannot receive source-specific offsets.")
        identities, descriptor_base = self._identity_tokens(count, indices, feature_descriptors, device)
        group_ids = self._group_ids(count, feature_descriptors, device)
        tokens = identities.unsqueeze(0).expand(batch_size, -1, -1).clone()

        if group_ids.any():
            tokens = tokens + self.group_embeddings(group_ids).unsqueeze(0)
        identity_and_group = tokens

        # Feature-set dropout hides only Level 3 feature slots. It is active
        # only during training and alters M_avail, never M_miss.
        feature_dropout = torch.zeros_like(available, dtype=torch.bool)
        if self.training and not is_unseen and feature_descriptors is not None and self.feature_set_dropout_rate > 0:
            specific = torch.tensor(
                [d.get("level") == 3 for d in feature_descriptors],
                device=device,
                dtype=torch.bool,
            )
            candidates = available.bool() & specific.unsqueeze(0)
            feature_dropout = (torch.rand(expected, device=device) < self.feature_set_dropout_rate) & candidates
            drop_all = torch.rand((batch_size, 1), device=device) < self.feature_set_dropout_all_probability
            feature_dropout = torch.where(drop_all, candidates, feature_dropout)
            available = torch.where(feature_dropout, torch.zeros_like(available), available)
            missing = torch.where(feature_dropout, torch.zeros_like(missing), missing)

        # Random value masking is a training-only measurement-missingness
        # augmentation. Structural availability remains unchanged.
        value_mask = torch.zeros_like(available, dtype=torch.bool)
        if self.training and self.value_masking_enabled and not is_unseen:
            eligible = available.bool() & (missing == 0) & (numeric_mask | categorical_mask).unsqueeze(0)
            value_mask = (torch.rand(expected, device=device) < self.value_masking_rate) & eligible
            missing = torch.where(value_mask, torch.ones_like(missing), missing)

        observed_numeric = numeric_mask.unsqueeze(0) & available.bool() & (missing == 0)
        if not torch.isfinite(numeric[observed_numeric]).all():
            raise ValueError("Observed numeric token values contain NaN or Inf.")
        safe_numeric = torch.where(observed_numeric, numeric, torch.zeros_like(numeric))
        numeric_input = torch.cat(
            [identities.unsqueeze(0).expand(batch_size, -1, -1), safe_numeric.unsqueeze(-1)],
            dim=-1,
        )
        numeric_features = self.numeric_projection(numeric_input)
        tokens = torch.where(observed_numeric.unsqueeze(-1), tokens + numeric_features, tokens)

        # Source categorical vocabularies are used when available. New feature
        # categories must be encoded as ID 0 (UNKNOWN) and use the shared token.
        for position in torch.where(categorical_mask)[0].tolist():
            ids = categorical[:, position]
            active = available[:, position].bool() & (missing[:, position] == 0)
            source_index = int(indices[position].item())
            slot = None
            if dataset in CATEGORICAL_CONFIG and source_index in CATEGORICAL_CONFIG[dataset]["positions"]:
                slot = CATEGORICAL_CONFIG[dataset]["positions"].index(source_index)
            if slot is None:
                if torch.any(ids[active] != 0):
                    raise ValueError("Unseen categorical features must use category ID 0 (UNKNOWN).")
                if active.any():
                    tokens[active, position] += self.unknown_category_embedding
            else:
                vocab_size = CATEGORICAL_CONFIG[dataset]["vocab_sizes"][slot]
                if torch.any(ids[active] < 0) or torch.any(ids[active] > vocab_size):
                    raise ValueError(f"Category ID outside source vocabulary at token {position}.")
                unknown = active & (ids == 0)
                known = active & (ids > 0)
                if known.any():
                    tokens[known, position] += self.categorical_embeddings[dataset][slot](ids[known])
                if unknown.any():
                    tokens[unknown, position] += self.unknown_category_embedding

        unavailable = available == 0
        measurement_missing = missing > 0
        tokens = torch.where(
            unavailable.unsqueeze(-1),
            identity_and_group + self.unavailable_embedding,
            tokens,
        )
        tokens = torch.where(
            measurement_missing.unsqueeze(-1),
            identity_and_group + self.missing_value_embedding,
            tokens,
        )
        mask_features = torch.stack([available, missing], dim=-1)
        tokens = tokens + self.mask_projection(mask_features)
        if not torch.isfinite(tokens).all():
            raise ValueError("Token representation contains NaN or Inf.")

        # Preserve a learned token for a known source feature that is
        # structurally unavailable. Extension padding (source index -1) and
        # features hidden by feature-set dropout are excluded entirely.
        structural_source = (indices >= 0).unsqueeze(0)
        pool_valid = available.bool() | (structural_source & ~feature_dropout)
        pool_weights = pool_valid.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * pool_weights).sum(dim=1) / pool_weights.sum(dim=1).clamp_min(1.0)
        shared = self.shared_encoder(pooled)
        context = self._context(descriptor_base, group_ids, feature_descriptors, available)
        dataset_specific = self.shared_context_adapter(pooled, context)
        return {
            "Z_s": shared,
            "Z_d": dataset_specific,
            "token_embeddings": tokens,
            "feature_context": context,
            "M_avail": available,
            "M_miss": missing,
            "value_mask_applied": value_mask,
            "feature_dropout_applied": feature_dropout,
        }
