"""Shared action-distribution metrics for eval and replay diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog

ActionDescriptorFn = Callable[..., dict[str, Any]]


def legal_indices_from_ids(*, legal_ids: np.ndarray, action_dim: int) -> np.ndarray:
    legal_array = np.asarray(legal_ids, dtype=np.int64)
    if legal_array.size == 0:
        return legal_array
    return legal_array[(legal_array >= 0) & (legal_array < int(action_dim))]


def masked_softmax(*, logits: np.ndarray, legal_indices: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(logits, dtype=np.float64)
    legal_logits = np.asarray(logits[legal_indices], dtype=np.float64)
    finite_mask = np.isfinite(legal_logits)
    if not bool(np.any(finite_mask)):
        probabilities[legal_indices] = 1.0 / float(legal_indices.size)
        return probabilities
    finite_logits = legal_logits[finite_mask]
    shifted = finite_logits - float(np.max(finite_logits))
    finite_probabilities = np.exp(shifted)
    finite_probabilities /= float(np.sum(finite_probabilities))
    probabilities[legal_indices[finite_mask]] = finite_probabilities
    return probabilities


def family_probability_masses(
    *,
    probabilities: np.ndarray,
    legal_indices: np.ndarray,
    action_catalog: ActionCatalog | None,
) -> dict[str, float]:
    if action_catalog is None:
        return {}
    masses: dict[str, float] = {}
    for action_index in legal_indices.tolist():
        family = action_catalog.decode(int(action_index)).family
        masses[family] = masses.get(family, 0.0) + float(probabilities[int(action_index)])
    return dict(sorted(masses.items(), key=lambda item: (-item[1], item[0])))


def top_action_payload(
    *,
    probabilities: np.ndarray,
    legal_indices: np.ndarray,
    action_catalog: ActionCatalog | None,
    descriptor_fn: ActionDescriptorFn | None = None,
    empty_error: str = "requires at least one legal action",
) -> dict[str, Any]:
    if legal_indices.size == 0:
        raise RuntimeError(empty_error)
    top_action = int(legal_indices[np.argmax(probabilities[legal_indices])])
    describe = simple_action_descriptor if descriptor_fn is None else descriptor_fn
    return {
        **describe(top_action, action_catalog=action_catalog),
        "probability": float(probabilities[top_action]),
    }


def simple_action_descriptor(action_id: int, *, action_catalog: ActionCatalog | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": int(action_id)}
    if action_catalog is None:
        return payload
    decoded = action_catalog.decode(int(action_id))
    payload["family"] = decoded.family
    return payload


def rank_of_action(*, probabilities: np.ndarray, legal_indices: np.ndarray, action_id: int) -> int:
    legal_probabilities = probabilities[legal_indices]
    sorted_indices = legal_indices[np.argsort(legal_probabilities)[::-1]]
    positions = np.flatnonzero(sorted_indices == int(action_id))
    if positions.size == 0:
        return int(legal_indices.shape[0]) + 1
    return int(positions[0]) + 1


def top_margin(*, values: np.ndarray, legal_indices: np.ndarray) -> float | None:
    if legal_indices.size < 2:
        return None
    legal_values = np.asarray(values[legal_indices], dtype=np.float64)
    if not np.all(np.isfinite(legal_values)):
        return None
    top_two = np.sort(legal_values)[-2:]
    return float(top_two[-1] - top_two[-2])


def gap_from_top_to_action(*, values: np.ndarray, legal_indices: np.ndarray, action_id: int) -> float | None:
    if legal_indices.size == 0 or not bool(np.any(legal_indices == int(action_id))):
        return None
    legal_values = np.asarray(values[legal_indices], dtype=np.float64)
    action_value = float(values[int(action_id)])
    if not np.all(np.isfinite(legal_values)) or not math.isfinite(action_value):
        return None
    return float(np.max(legal_values) - action_value)


def same_family_margin_to_action(
    *,
    values: np.ndarray,
    legal_indices: np.ndarray,
    action_id: int,
    action_catalog: ActionCatalog | None,
) -> float | None:
    if action_catalog is None or legal_indices.size == 0 or not bool(np.any(legal_indices == int(action_id))):
        return None
    action_value = float(values[int(action_id)])
    if not math.isfinite(action_value):
        return None
    target_family = action_catalog.decode(int(action_id)).family
    same_family_legal_indices = np.asarray(
        [
            int(legal_id)
            for legal_id in legal_indices.tolist()
            if int(legal_id) != int(action_id) and action_catalog.decode(int(legal_id)).family == target_family
        ],
        dtype=np.int64,
    )
    if same_family_legal_indices.size == 0:
        return None
    competitor_values = np.asarray(values[same_family_legal_indices], dtype=np.float64)
    if not np.all(np.isfinite(competitor_values)):
        return None
    return float(action_value - np.max(competitor_values))


def percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite_values = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite_values.size == 0:
        return {"count": 0, "mean": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "p10": float(np.percentile(finite_values, 10)),
        "p25": float(np.percentile(finite_values, 25)),
        "p50": float(np.percentile(finite_values, 50)),
        "p75": float(np.percentile(finite_values, 75)),
        "p90": float(np.percentile(finite_values, 90)),
    }


def max_non_reference_probability(
    *,
    probabilities: np.ndarray,
    legal_indices: np.ndarray,
    reference_action: int,
) -> float:
    non_reference = legal_indices[legal_indices != int(reference_action)]
    if non_reference.size == 0:
        return 0.0
    return float(np.max(probabilities[non_reference]))


def append_optional(values: list[float], value: float | None) -> None:
    if value is not None and math.isfinite(float(value)):
        values.append(float(value))


__all__ = [
    "append_optional",
    "family_probability_masses",
    "gap_from_top_to_action",
    "legal_indices_from_ids",
    "masked_softmax",
    "max_non_reference_probability",
    "percentile_summary",
    "rank_of_action",
    "same_family_margin_to_action",
    "simple_action_descriptor",
    "top_action_payload",
    "top_margin",
]
