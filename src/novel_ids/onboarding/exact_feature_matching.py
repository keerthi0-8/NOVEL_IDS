"""Collision-safe exact matching for raw feature rows."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd


def _typed_value(value: Any) -> tuple[str, Any]:
    """Return a stable, non-rounded key while preserving value type."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return ("missing", "")
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return ("missing", "")
        return ("float", number.hex())
    if isinstance(value, Decimal):
        return ("decimal", str(value))
    if isinstance(value, str):
        return ("str", value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ("missing", "")
    except (TypeError, ValueError):
        pass
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def exact_feature_key(values: Any) -> tuple[tuple[str, Any], ...]:
    """Build an exact typed key for one row; numeric values are never rounded."""
    return tuple(_typed_value(value) for value in values)


def exact_feature_overlap_groups(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    candidate_hasher: Callable[[tuple[tuple[str, Any], ...]], Any] | None = None,
) -> list[dict[str, list[int]]]:
    """Return every shared exact feature group with positional occurrences.

    The DataFrames must have the same feature columns in the same order.
    A hash may optionally narrow candidate buckets, but the complete typed key
    is always compared before rows are declared equal. Repeated occurrences
    remain listed on their respective side.
    """
    if list(left.columns) != list(right.columns):
        raise ValueError("Exact feature comparison requires identical ordered columns.")

    key_function = candidate_hasher or (lambda key: key)
    left_positions: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
    right_positions: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
    left_buckets: dict[Any, set[tuple[tuple[str, Any], ...]]] = defaultdict(set)
    right_buckets: dict[Any, set[tuple[str, Any], ...]] = defaultdict(set)
    for position, row in enumerate(left.itertuples(index=False, name=None)):
        key = exact_feature_key(row)
        left_positions[key].append(position)
        bucket = key_function(key)
        left_buckets[bucket].add(key)
    for position, row in enumerate(right.itertuples(index=False, name=None)):
        key = exact_feature_key(row)
        right_positions[key].append(position)
        bucket = key_function(key)
        right_buckets[bucket].add(key)

    # Recheck full keys inside each candidate bucket; collisions alone never
    # establish equality.
    common = {
        left_key
        for bucket, left_keys in left_buckets.items()
        for left_key in left_keys
        for right_key in right_buckets.get(bucket, ())
        if left_key == right_key
    }
    groups = [
        {
            "left_positions": left_positions[key],
            "right_positions": right_positions[key],
        }
        for key in common
    ]
    groups.sort(key=lambda group: (group["left_positions"][0], group["right_positions"][0]))
    return groups
