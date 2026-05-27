"""Cached simulator card-table helpers for structured policy models."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

_TRAIT_HASH_DIM = 8


@lru_cache(maxsize=1)
def cached_runtime_card_table() -> dict[str, Any] | None:
    """Load the simulator card table once from the local `weiss_sim` package."""

    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ImportError:
        return None
    exporter = getattr(weiss_sim, "export_card_table", None)
    if not callable(exporter):
        return None
    payload = exporter()
    return dict(payload) if isinstance(payload, Mapping) else None


def card_feature_table(
    *,
    card_table: Mapping[str, Any] | None,
    vocab_size: int,
) -> np.ndarray:
    """Build a dense static feature table indexed by simulator card id."""

    rows_obj = [] if card_table is None else card_table.get("rows", [])
    rows = [dict(item) for item in rows_obj] if isinstance(rows_obj, list) else []
    colors = sorted({str(row.get("color", "")).strip().lower() for row in rows if str(row.get("color", "")).strip()})
    card_types = sorted(
        {str(row.get("card_type", "")).strip().lower() for row in rows if str(row.get("card_type", "")).strip()}
    )
    color_index = {name: idx for idx, name in enumerate(colors)}
    type_index = {name: idx for idx, name in enumerate(card_types)}
    feature_dim = 4 + len(color_index) + len(type_index) + _TRAIT_HASH_DIM
    table = np.zeros((int(vocab_size), int(feature_dim)), dtype=np.float32)
    if not rows:
        return table

    for row in rows:
        try:
            card_id = int(row.get("card_id", -1))
        except (TypeError, ValueError):
            continue
        if card_id < 0 or card_id >= int(vocab_size):
            continue
        offset = 0
        table[card_id, offset + 0] = _normalize_float(row.get("level"), scale=8.0)
        table[card_id, offset + 1] = _normalize_float(row.get("cost"), scale=8.0)
        table[card_id, offset + 2] = _normalize_float(row.get("power"), scale=20000.0)
        table[card_id, offset + 3] = _normalize_float(row.get("soul"), scale=4.0)
        offset += 4
        color_name = str(row.get("color", "")).strip().lower()
        if color_name in color_index:
            table[card_id, offset + color_index[color_name]] = 1.0
        offset += len(color_index)
        type_name = str(row.get("card_type", "")).strip().lower()
        if type_name in type_index:
            table[card_id, offset + type_index[type_name]] = 1.0
        offset += len(type_index)
        for trait in _coerce_traits(row.get("traits")):
            bucket = hash(trait) % _TRAIT_HASH_DIM
            table[card_id, offset + bucket] += 1.0
        trait_slice = table[card_id, offset : offset + _TRAIT_HASH_DIM]
        trait_norm = float(np.linalg.norm(trait_slice))
        if trait_norm > 0.0:
            trait_slice /= trait_norm
    return table


def _normalize_float(value: Any, *, scale: float) -> float:
    try:
        return float(value) / float(scale)
    except (TypeError, ValueError):
        return 0.0


def _coerce_traits(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    traits: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            traits.append(text)
    return traits


__all__ = ["cached_runtime_card_table", "card_feature_table"]
