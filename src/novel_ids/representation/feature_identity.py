
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


TEXT_DIM = 512
OUTPUT_DIM = 128

STAT_KEYS = [
    "q01",
    "q25",
    "median",
    "q75",
    "q99",
    "mean",
    "std",
    "skewness",
    "zero_fraction",
    "distinct_count",
    "missing_fraction",
    "is_binary",
    "is_integer",
]

DATA_TYPES = ["continuous", "symbolic", "binary", "unknown"]
LEVELS = [1, 2, 3]
CONFIDENCE = ["high", "medium", "low", "unassessed"]


class FeatureIdentityEncoder(nn.Module):
    """
    Generates learnable 128-dimensional feature identities.

    Level 1 with semantic_id:
        Uses canonical identity shared by equivalent features.

    Level 1 without semantic_id:
        Uses legacy descriptor information for compatibility.

    Level 2:
        Retains individual feature descriptions and semantic groups.

    Level 3:
        Retains dataset-specific feature descriptions.
    """

    def __init__(
        self,
        output_dim: int = OUTPUT_DIM,
        text_dim: int = TEXT_DIM,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.output_dim = output_dim

        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            n_features=text_dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )

        self.tfidf = TfidfTransformer()
        self.fitted = False

        metadata_dim = (
            len(STAT_KEYS)
            + len(DATA_TYPES)
            + len(LEVELS)
            + len(CONFIDENCE)
        )

        self.projection = nn.Sequential(
            nn.Linear(text_dim + metadata_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    @staticmethod
    def _has_canonical_identity(
        descriptor: dict[str, Any],
    ) -> bool:
        return (
            descriptor.get("level") == 1
            and bool(descriptor.get("semantic_id"))
        )

    @staticmethod
    def _text(descriptor: dict[str, Any]) -> str:
        """Create canonical or descriptor-based text."""

        if FeatureIdentityEncoder._has_canonical_identity(
            descriptor
        ):
            return str(
                descriptor["semantic_id"]
            ).strip().lower()

        fields = [
            descriptor.get("original_feature_name", ""),
            descriptor.get("metadata_feature_name", ""),
            descriptor.get("semantic_group", ""),
            descriptor.get("relationship", ""),
            descriptor.get("description", ""),
            descriptor.get("data_type", "unknown"),
            descriptor.get("unit", ""),
        ]

        return " ".join(
            str(value)
            for value in fields
            if value is not None and str(value).strip()
        )

    @staticmethod
    def _numeric_metadata(
        descriptor: dict[str, Any],
    ) -> np.ndarray:
        """Convert feature metadata into numeric inputs."""

        canonical = FeatureIdentityEncoder._has_canonical_identity(
            descriptor
        )

        if canonical:
            # Equivalent Level 1 features receive identical metadata.
            stats = {}
            data_type = "unknown"
            confidence = "unassessed"

        else:
            # Preserve individual metadata for legacy, Level 2 and Level 3.
            stats = descriptor.get("statistics") or {}

            data_type = str(
                descriptor.get("data_type", "unknown")
            ).lower()

            confidence = str(
                descriptor.get("confidence", "unassessed")
            ).lower()

        values = []

        for key in STAT_KEYS:
            value = stats.get(key, 0.0)

            if value is None:
                value = 0.0

            value = float(value)

            if not np.isfinite(value):
                value = 0.0

            if key not in {
                "zero_fraction",
                "missing_fraction",
                "is_binary",
                "is_integer",
            }:
                value = np.sign(value) * np.log1p(abs(value))

            values.append(value)

        if data_type not in DATA_TYPES:
            data_type = "unknown"

        values.extend(
            [
                float(data_type == item)
                for item in DATA_TYPES
            ]
        )

        level = descriptor.get("level", 3)

        values.extend(
            [
                float(level == item)
                for item in LEVELS
            ]
        )

        if confidence not in CONFIDENCE:
            confidence = "unassessed"

        values.extend(
            [
                float(confidence == item)
                for item in CONFIDENCE
            ]
        )

        result = np.asarray(
            values,
            dtype=np.float32,
        )

        if not np.isfinite(result).all():
            raise ValueError(
                "Feature metadata contains non-finite values."
            )

        return result

    def fit(self, source_training_descriptors):
        """Fit text weighting on source-training descriptors only.

        Descriptor statistics must already have been computed from source
        training partitions. Labels and validation/test/unseen records are
        neither accepted nor consulted here.
        """

        if not source_training_descriptors:
            raise ValueError(
                "Source-training descriptors cannot be empty."
            )

        texts = [
            self._text(item)
            for item in source_training_descriptors
        ]

        hashed = self.vectorizer.transform(texts)

        self.tfidf.fit(hashed)

        self.fitted = True

        return self

    def _descriptor_features(self, descriptors):
        if not self.fitted:
            raise RuntimeError(
                "Call fit() on source-training descriptors first."
            )
        if not descriptors:
            raise ValueError("Descriptors cannot be empty.")

        texts = [self._text(item) for item in descriptors]
        text_features = self.tfidf.transform(
            self.vectorizer.transform(texts)
        ).toarray()
        metadata_features = np.stack(
            [self._numeric_metadata(item) for item in descriptors]
        )
        return np.concatenate(
            [text_features, metadata_features], axis=1
        ).astype(np.float32, copy=False)

    def map_unseen_descriptor(
        self,
        descriptor: dict[str, Any],
        source_descriptors,
        similarity_threshold: float,
    ) -> dict[str, Any]:
        """Assign an unseen feature to a reviewed Level 2 group or Level 3.

        Similarity never promotes an unseen feature to Level 1. A caller must
        provide an explicit threshold; all comparisons use descriptor data,
        never labels.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1].")
        if not source_descriptors:
            raise ValueError("source_descriptors cannot be empty.")

        candidates = [
            item for item in source_descriptors
            if item.get("level") == 2
            and item.get("semantic_group")
            and item.get("relationship") in {"group_only", "partial"}
            and item.get("review_status") == "approved"
        ]
        result = dict(descriptor)
        if not candidates:
            return self._as_level_three(result, "No reviewed Level 2 groups.")

        matrix = self._descriptor_features(candidates + [descriptor])
        candidate_vectors = matrix[:-1]
        query = matrix[-1]
        denom = np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query)
        similarities = np.divide(
            candidate_vectors @ query,
            denom,
            out=np.zeros(len(candidates), dtype=np.float32),
            where=denom > 0,
        )
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        if best_similarity < similarity_threshold:
            return self._as_level_three(
                result,
                f"Best reviewed-group descriptor similarity {best_similarity:.4f} "
                f"did not meet threshold {similarity_threshold:.4f}.",
            )

        source = candidates[best_index]
        result.update(
            {
                "level": 2,
                "classification": "semantic_group_candidate",
                "semantic_id": None,
                "semantic_group": source["semantic_group"],
                "relationship": "group_only",
                "confidence": "medium",
                "review_status": "approved",
                "similarity": best_similarity,
                "evidence": (
                    "Assigned to reviewed semantic group by descriptor "
                    f"similarity {best_similarity:.4f}; no Level 1 equivalence inferred."
                ),
            }
        )
        return result

    @staticmethod
    def _as_level_three(descriptor, evidence):
        descriptor.update(
            {
                "level": 3,
                "classification": "dataset_specific",
                "semantic_id": None,
                "semantic_group": None,
                "relationship": "dataset_specific",
                "confidence": "unassessed",
                "review_status": "dataset_specific",
                "evidence": evidence,
            }
        )
        return descriptor

    def encode(self, descriptors):
        """Generate 128-dimensional feature identity vectors."""

        if not self.fitted:
            raise RuntimeError(
                "Call fit() on source-training descriptors first."
            )

        if not descriptors:
            raise ValueError(
                "Descriptors cannot be empty."
            )

        combined = self._descriptor_features(descriptors)

        inputs = torch.as_tensor(
            combined,
            dtype=torch.float32,
            device=self.projection[0].weight.device,
        )

        identities = self.projection(inputs)

        if not torch.isfinite(identities).all():
            raise ValueError(
                "Identity encoder produced NaN or Inf."
            )

        return identities

    def forward(self, descriptors):
        return self.encode(descriptors)
