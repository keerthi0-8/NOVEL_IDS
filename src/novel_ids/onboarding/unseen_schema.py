"""Build variable-length model inputs from an unlabeled onboarding frame.

Unseen numeric features use deterministic robust normalization fitted only to
the finite values in the supplied unlabeled onboarding frame. This does not
touch fitted source preprocessors or fit model parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from novel_ids.representation.dual_encoder import TOKEN_COUNT
from novel_ids.representation.feature_identity import FeatureIdentityEncoder


class UnseenNumericNormalizer:
    """Median/IQR normalization state for one unlabeled unseen feature set."""

    method = "median_iqr"
    fit_scope = "unseen_unlabeled_available_finite_values_only"

    def __init__(self) -> None:
        self.feature_names: list[str] | None = None
        self.statistics: dict[str, dict[str, float | int]] | None = None

    def fit(self, frame: pd.DataFrame) -> "UnseenNumericNormalizer":
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Unseen normalization requires non-empty numeric data.")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise TypeError("UnseenNumericNormalizer accepts numeric features only.")

        self.feature_names = list(frame.columns)
        self.statistics = {}
        for name in self.feature_names:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                median = float(np.median(finite))
                q25, q75 = np.quantile(finite, [0.25, 0.75], method="linear")
                iqr = float(q75 - q25)
                scale = iqr if np.isfinite(iqr) and iqr > 0 else 1.0
            else:
                # No measurement can define a center; placeholders stay masked.
                median, iqr, scale = 0.0, 0.0, 1.0
            self.statistics[name] = {
                "median": median,
                "q25": float(q25) if finite.size else 0.0,
                "q75": float(q75) if finite.size else 0.0,
                "iqr": iqr,
                "scale": scale,
                "finite_count": int(finite.size),
            }
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.statistics is None or self.feature_names is None:
            raise RuntimeError("Call fit() before transform().")
        if list(frame.columns) != self.feature_names:
            raise ValueError("Unseen numeric feature names/order differ from fitted state.")
        raw = frame.to_numpy(dtype=np.float64, copy=True)
        missing = ~np.isfinite(raw)
        result = np.zeros(raw.shape, dtype=np.float64)
        for index, name in enumerate(self.feature_names):
            state = self.statistics[name]
            observed = ~missing[:, index]
            result[observed, index] = (
                raw[observed, index] - float(state["median"])
            ) / float(state["scale"])
        result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Unseen normalization produced NaN or Inf values.")
        return result, missing.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.statistics is None or self.feature_names is None:
            raise RuntimeError("Cannot serialize an unfitted unseen normalizer.")
        return {
            "method": self.method,
            "fit_scope": self.fit_scope,
            "feature_names": list(self.feature_names),
            "statistics": {name: dict(self.statistics[name]) for name in self.feature_names},
        }


def _descriptor(name: str, values: pd.Series) -> dict[str, Any]:
    numeric = pd.api.types.is_numeric_dtype(values.dtype)
    if numeric:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = clean.dropna()
        count = int(valid.nunique())
        stats = {
            "q01": float(valid.quantile(.01)) if len(valid) else 0.0,
            "q25": float(valid.quantile(.25)) if len(valid) else 0.0,
            "median": float(valid.median()) if len(valid) else 0.0,
            "q75": float(valid.quantile(.75)) if len(valid) else 0.0,
            "q99": float(valid.quantile(.99)) if len(valid) else 0.0,
            "mean": float(valid.mean()) if len(valid) else 0.0,
            "std": float(valid.std()) if len(valid) else 0.0,
            "skewness": float(valid.skew()) if len(valid) > 2 and np.isfinite(valid.skew()) else 0.0,
            "zero_fraction": float((valid == 0).mean()) if len(valid) else 0.0,
            "distinct_count": count,
            "missing_fraction": float(clean.isna().mean()),
            "is_binary": count <= 2,
            "is_integer": bool(len(valid) and np.allclose(valid, np.round(valid))),
        }
        data_type = "binary" if count <= 2 else "continuous"
    else:
        stats = {"distinct_count": int(values.nunique(dropna=True)), "missing_fraction": float(values.isna().mean())}
        data_type = "symbolic"
    return {
        "original_feature_name": str(name),
        "metadata_feature_name": str(name),
        "description": f"Unlabeled onboarding descriptor for {name}",
        "data_type": data_type,
        "statistics": stats,
    }


def build_unseen_model_inputs(
    frame: pd.DataFrame,
    source_descriptors: list[dict[str, Any]],
    identity_encoder: FeatureIdentityEncoder,
    *,
    similarity_threshold: float = 1.0,
    label_columns: tuple[str, ...] | list[str] = (),
    normalization_frame: pd.DataFrame | None = None,
    matching_descriptors: list[dict[str, Any]] | None = None,
    external_metadata: dict[str, dict[str, Any]] | None = None,
    normalization_mode: str = "option_a_target",
) -> dict[str, Any]:
    """Create 78 unavailable source slots followed by frame-ordered features.

    ``label_columns`` names outcome columns to exclude before onboarding. Only
    feature values are inspected. The unseen normalizer and descriptor
    statistics use a separate unlabeled calibration frame when supplied;
    otherwise they use the unlabeled evaluation feature frame. No source
    scaler, labels, predictions, outcomes, feedback, or model parameters are used.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not len(frame.columns):
        raise ValueError("An unlabeled, non-empty feature DataFrame is required.")
    if len(source_descriptors) != TOKEN_COUNT:
        raise ValueError("Exactly 78 source descriptors are required.")
    if not identity_encoder.fitted:
        raise ValueError("A source-training-fitted feature identity encoder is required.")

    excluded = set(label_columns)
    feature_frame = frame.drop(columns=[name for name in frame.columns if name in excluded]).copy()
    if feature_frame.empty:
        raise ValueError("At least one unlabeled feature column is required.")
    statistics_frame = (normalization_frame.copy() if normalization_frame is not None else feature_frame.copy())
    if list(statistics_frame.columns) != list(feature_frame.columns):
        raise ValueError("Unlabeled normalization frame columns/order must match evaluation features exactly.")
    external_metadata = external_metadata or {}
    if normalization_mode not in {"option_a_target", "option_b_strict_source_level1"}:
        raise ValueError("Unknown unseen normalization mode.")
    def apply_declared_transforms(source: pd.DataFrame) -> pd.DataFrame:
        transformed = source.copy()
        protocol_names = {1: "icmp", 6: "tcp", 17: "udp", 47: "gre", 50: "esp", 58: "ipv6-icmp"}
        for feature_name, metadata in external_metadata.items():
            if feature_name not in transformed.columns:
                continue
            transform = metadata.get("value_transform")
            if transform == "microseconds_to_seconds":
                transformed[feature_name] = pd.to_numeric(transformed[feature_name], errors="coerce") / 1_000_000.0
            elif transform == "iana_protocol_to_name":
                numeric = pd.to_numeric(transformed[feature_name], errors="coerce")
                transformed[feature_name] = numeric.map(
                    lambda value: protocol_names.get(int(value), f"protocol_{int(value)}")
                    if pd.notna(value) else "protocol_unknown"
                ).astype("string")
        return transformed
    feature_frame = apply_declared_transforms(feature_frame)
    statistics_frame = apply_declared_transforms(statistics_frame)
    matching_descriptors = matching_descriptors or source_descriptors
    descriptors = list(source_descriptors)
    extension_descriptors = []
    n, extra = len(feature_frame), len(feature_frame.columns)
    numeric = np.zeros((n, TOKEN_COUNT + extra), dtype=np.float32)
    categorical = np.zeros((n, TOKEN_COUNT + extra), dtype=np.int64)
    available = np.zeros((n, TOKEN_COUNT + extra), dtype=np.float32)
    missing = np.zeros_like(available)
    numeric_mask = np.zeros(TOKEN_COUNT + extra, dtype=bool)
    categorical_mask = np.zeros(TOKEN_COUNT + extra, dtype=bool)

    # External schema metadata is authoritative for declared numeric types.
    # CSV chunks can infer a numeric column as object when a later row contains a
    # malformed token; coerce both the current chunk and the unlabeled calibration
    # frame consistently before fitting/applying target-only normalization.
    numeric_names = []
    for name in feature_frame.columns:
        metadata_type = str(external_metadata.get(str(name), {}).get("data_type", "")).casefold()
        if pd.api.types.is_numeric_dtype(feature_frame[name].dtype) or metadata_type in {"numeric", "continuous", "integer", "binary"}:
            numeric_names.append(name)
    for name in numeric_names:
        feature_frame[name] = pd.to_numeric(feature_frame[name], errors="coerce")
        statistics_frame[name] = pd.to_numeric(statistics_frame[name], errors="coerce")
    normalizer = UnseenNumericNormalizer()
    if numeric_names:
        normalizer.fit(statistics_frame[numeric_names])
        if normalization_frame is not None:
            normalizer.fit_scope = "unseen_unlabeled_uniform_calibration_sample_only"
        normalized_numeric, numeric_missing = normalizer.transform(feature_frame[numeric_names])
        numeric_position = {name: index for index, name in enumerate(numeric_names)}
        source_normalization = {}
        if normalization_mode == "option_b_strict_source_level1":
            for feature_name in numeric_names:
                mapping = external_metadata.get(str(feature_name), {}).get("mapping", {})
                if int(mapping.get("level", 3)) != 1:
                    continue
                semantic_id = mapping.get("target_semantic_id")
                references = [item.get("statistics", {}) for item in matching_descriptors
                              if item.get("level") == 1 and item.get("semantic_id") == semantic_id
                              and item.get("review_status") == "approved"]
                valid = [item for item in references
                         if all(np.isfinite(float(item.get(k, np.nan))) for k in ("median", "q25", "q75"))]
                if not valid:
                    raise ValueError(f"Option B has no source-training statistics for {feature_name}.")
                median = float(np.mean([float(item["median"]) for item in valid]))
                iqr = float(np.mean([float(item["q75"])-float(item["q25"]) for item in valid]))
                scale = iqr if np.isfinite(iqr) and iqr > 0 else 1.0
                raw = pd.to_numeric(feature_frame[feature_name], errors="coerce").to_numpy(dtype=np.float64)
                observed = np.isfinite(raw)
                position = numeric_position[feature_name]
                normalized_numeric[observed, position] = ((raw[observed]-median)/scale).astype(np.float32)
                source_normalization[feature_name] = {
                    "semantic_id": semantic_id, "source_reference_count": len(valid),
                    "median_equal_dataset_mean": median, "iqr_equal_dataset_mean": iqr,
                    "scale": scale, "source": "audited source-training feature descriptors only",
                }
            normalizer.fit_scope = "target_unlabeled_statistics_for_non_level1; source_training_statistics_for_level1"
        else:
            source_normalization = {}
    else:
        normalized_numeric = np.empty((n, 0), dtype=np.float32)
        numeric_missing = np.empty((n, 0), dtype=np.float32)
        numeric_position = {}

    for offset, name in enumerate(feature_frame.columns):
        position = TOKEN_COUNT + offset
        values = feature_frame[name]
        descriptor = _descriptor(str(name), statistics_frame[name])
        metadata = external_metadata.get(str(name), {})
        mapping = metadata.get("mapping", {}) if metadata else {}
        if metadata:
            descriptor.update({
                "description": str(metadata.get("description", descriptor["description"])),
                "unit": metadata.get("unit"),
                "network_meaning": metadata.get("network_meaning"),
                "data_type": str(metadata.get("data_type", descriptor["data_type"])).lower(),
                "metadata_feature_name": str(name),
            })
        if mapping and int(mapping.get("level", 3)) == 1:
            semantic_id = mapping.get("target_semantic_id")
            candidates = [item for item in matching_descriptors
                          if item.get("level") == 1 and item.get("semantic_id") == semantic_id
                          and item.get("review_status") == "approved"]
            if candidates and mapping.get("unit_compatible") is True and mapping.get("type_compatible") is True:
                query = dict(descriptor)
                query.update({"level": 1, "semantic_id": semantic_id, "semantic_group": None,
                              "relationship": "equivalent", "confidence": "high",
                              "review_status": "approved"})
                matrix = identity_encoder._descriptor_features([candidates[0], query])
                denom = np.linalg.norm(matrix[0]) * np.linalg.norm(matrix[1])
                similarity = float(np.dot(matrix[0], matrix[1]) / denom) if denom > 0 else 0.0
                if similarity >= float(mapping.get("similarity_threshold", 0.15)):
                    query["similarity"] = similarity
                    query["evidence"] = str(mapping.get("evidence", "Expert metadata semantic identity; unit/type compatibility and descriptor similarity verified."))
                    descriptor = query
                else:
                    descriptor.update({"level": 3, "semantic_id": None, "semantic_group": None,
                                       "relationship": "dataset_specific", "confidence": "unassessed",
                                       "review_status": "dataset_specific",
                                       "evidence": f"Level 1 metadata candidate rejected: descriptor similarity {similarity:.4f} below threshold."})
            else:
                descriptor.update({"level": 3, "semantic_id": None, "semantic_group": None,
                                   "relationship": "dataset_specific", "confidence": "unassessed",
                                   "review_status": "dataset_specific",
                                   "evidence": "Level 1 metadata candidate rejected: missing approved source identity or unit/type compatibility."})
        elif mapping and int(mapping.get("level", 3)) == 2:
            group = mapping.get("semantic_group")
            candidates = [item for item in matching_descriptors
                          if item.get("level") == 2 and item.get("semantic_group") == group
                          and item.get("relationship") in {"group_only", "partial"}
                          and item.get("review_status") == "approved"]
            if candidates and mapping.get("unit_compatible") is True and mapping.get("type_compatible") is True:
                query = dict(descriptor)
                query.update({"level": 2, "semantic_id": None, "semantic_group": group,
                              "relationship": "group_only", "confidence": "medium",
                              "review_status": "approved"})
                matrix = identity_encoder._descriptor_features(candidates + [query])
                q = matrix[-1]; cand = matrix[:-1]
                denom = np.linalg.norm(cand, axis=1) * np.linalg.norm(q)
                sims = np.divide(cand @ q, denom, out=np.zeros(len(candidates)), where=denom > 0)
                best = int(np.argmax(sims)); similarity = float(sims[best])
                if similarity >= float(mapping.get("similarity_threshold", similarity_threshold)):
                    query["similarity"] = similarity
                    query["evidence"] = str(mapping.get("evidence", "Related group only; descriptor similarity and unit/type compatibility verified; no equivalence asserted."))
                    descriptor = query
                else:
                    descriptor.update({"level": 3, "semantic_id": None, "semantic_group": None,
                                       "relationship": "dataset_specific", "confidence": "unassessed",
                                       "review_status": "dataset_specific",
                                       "evidence": f"Group candidate rejected: descriptor similarity {similarity:.4f} below threshold."})
            else:
                descriptor.update({"level": 3, "semantic_id": None, "semantic_group": None,
                                   "relationship": "dataset_specific", "confidence": "unassessed",
                                   "review_status": "dataset_specific",
                                   "evidence": "Group-only candidate rejected: no approved source group or unit/type compatibility."})
        elif mapping and int(mapping.get("level", 3)) == 3:
            descriptor.update({"level": 3, "semantic_id": None, "semantic_group": None,
                               "relationship": "dataset_specific", "confidence": "unassessed",
                               "review_status": "dataset_specific",
                               "evidence": str(mapping.get("evidence", "No approved source equivalent or related group; retained as dataset-specific."))})
        else:
            exact = [item for item in matching_descriptors
                     if item.get("original_feature_name") == str(name)
                     and item.get("review_status") == "approved"
                     and item.get("level") in {1, 2}]
            if exact:
                source = exact[0]
                if source.get("level") == 1 and source.get("semantic_id"):
                    descriptor.update({
                        "level": 1, "semantic_id": source["semantic_id"], "semantic_group": None,
                        "relationship": "equivalent", "confidence": "high",
                        "review_status": "approved", "similarity": 1.0,
                        "evidence": "Exact source-training feature-name match to an approved Level-1 semantic identity.",
                    })
                elif source.get("level") == 2 and source.get("semantic_group"):
                    descriptor.update({
                        "level": 2, "semantic_id": None, "semantic_group": source["semantic_group"],
                        "relationship": "group_only", "confidence": "medium",
                        "review_status": "approved", "similarity": 1.0,
                        "evidence": "Exact source-training feature-name match; retained as group_only, never promoted to equivalence.",
                    })
                else:
                    descriptor = identity_encoder.map_unseen_descriptor(
                        descriptor, matching_descriptors, similarity_threshold
                    )
            else:
                descriptor = identity_encoder.map_unseen_descriptor(
                    descriptor, matching_descriptors, similarity_threshold
                )
        extension_descriptors.append(descriptor)
        available[:, position] = 1.0
        if pd.api.types.is_numeric_dtype(values.dtype):
            numeric[:, position] = normalized_numeric[:, numeric_position[name]]
            missing[:, position] = numeric_missing[:, numeric_position[name]]
            numeric_mask[position] = True
        else:
            # All categories are unseen because no source vocabulary exists.
            categorical[:, position] = 0
            missing[:, position] = values.isna().to_numpy(dtype=np.float32)
            categorical_mask[position] = True
    descriptors.extend(extension_descriptors)
    return {
        "numeric_values": numeric,
        "categorical_values": categorical,
        "M_avail": available,
        "M_miss": missing,
        "numeric_token_mask": numeric_mask,
        "categorical_token_mask": categorical_mask,
        # This path is strictly zero-shot inference: even unavailable source
        # padding slots receive no source-specific trainable offsets.
        "source_feature_indices": np.full(TOKEN_COUNT + extra, -1, dtype=np.int64),
        "feature_descriptors": descriptors,
        "is_unseen": True,
        "onboarded_feature_names": list(feature_frame.columns),
        "declared_feature_transforms": {
            name: metadata.get("value_transform")
            for name, metadata in external_metadata.items()
            if name in feature_frame.columns and metadata.get("value_transform")
        },
        "normalization_mode": normalization_mode,
        "source_training_level1_normalization": source_normalization if numeric_names else {},
        "unseen_numeric_normalization": normalizer.to_dict() if numeric_names else {
            "method": "median_iqr",
            "fit_scope": UnseenNumericNormalizer.fit_scope,
            "feature_names": [],
            "statistics": {},
        },
    }
