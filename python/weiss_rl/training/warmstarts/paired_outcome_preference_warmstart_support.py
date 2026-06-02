"""Reusable helpers for paired-outcome preference warmstart orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weiss_rl.training.warmstarts.warmstart_replay_support import (
    _initial_hidden_state,
    _opponent_context_indices_for_episodes,
    _source_opponent_policy_ids_by_episode,
)


def _parse_pair_weights(raw_values: Sequence[str] | None) -> dict[int, float]:
    weights: dict[int, float] = {}
    for raw_value in raw_values or []:
        text = str(raw_value).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("--pair-weight values must use PAIR_ID=WEIGHT")
        raw_pair_id, raw_weight = text.split("=", 1)
        try:
            pair_id = int(raw_pair_id.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --pair-weight pair id: {raw_pair_id!r}") from exc
        if pair_id < 0:
            raise ValueError("--pair-weight pair id must be nonnegative")
        try:
            weight = float(raw_weight.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --pair-weight value for pair {pair_id}: {raw_weight!r}") from exc
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("--pair-weight weights must be finite and positive")
        weights[pair_id] = weight
    return weights


def _parse_pair_role_selectors(raw_values: Sequence[str] | None) -> tuple[tuple[int, int | None], ...]:
    selectors: list[tuple[int, int | None]] = []
    for raw_value in raw_values or []:
        text = str(raw_value).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError("retention pair-role values must use PAIR_ID:ROLE")
        raw_pair_id, raw_role = text.split(":", 1)
        try:
            pair_id = int(raw_pair_id.strip())
        except ValueError as exc:
            raise ValueError(f"invalid retention pair id: {raw_pair_id!r}") from exc
        if pair_id < 0:
            raise ValueError("retention pair id must be nonnegative")
        role_text = raw_role.strip().lower()
        if role_text in {"all", "*"}:
            role: int | None = None
        elif role_text in {"preferred", "pref", "1"}:
            role = 1
        elif role_text in {"rejected", "rej", "0"}:
            role = 0
        else:
            raise ValueError("retention role must be preferred, rejected, or all")
        selector = (pair_id, role)
        if selector not in selectors:
            selectors.append(selector)
    return tuple(selectors)


def _serialize_pair_role_selectors(selectors: Sequence[tuple[int, int | None]]) -> list[str]:
    role_names = {None: "all", 1: "preferred", 0: "rejected"}
    return [f"{int(pair_id)}:{role_names[role]}" for pair_id, role in selectors]


def _preference_pair_weight_matrix(preference_pair_ids: Any, pair_weights: Mapping[int, float]) -> np.ndarray:
    if preference_pair_ids is None:
        raise ValueError("batch is missing preference_pair_id; cannot apply --pair-weight")
    pair_ids = np.asarray(preference_pair_ids)
    weights = np.ones(pair_ids.shape, dtype=np.float32)
    for pair_id, weight in pair_weights.items():
        weights[pair_ids == int(pair_id)] = float(weight)
    return weights


def _preference_pair_role_mask(
    preference_pair_ids: Any,
    preference_roles: Any,
    selectors: Sequence[tuple[int, int | None]],
) -> np.ndarray:
    if preference_pair_ids is None:
        raise ValueError("batch is missing preference_pair_id; cannot apply retention pair-role selectors")
    if preference_roles is None:
        raise ValueError("batch is missing preference_role; cannot apply retention pair-role selectors")
    pair_ids = np.asarray(preference_pair_ids)
    roles = np.asarray(preference_roles)
    if pair_ids.shape != roles.shape:
        raise ValueError("preference_pair_id and preference_role must have the same shape")
    mask = np.zeros(pair_ids.shape, dtype=np.float32)
    for pair_id, role in selectors:
        selector_mask = pair_ids == int(pair_id)
        if role is not None:
            selector_mask = selector_mask & (roles == int(role))
        mask[selector_mask] = 1.0
    return mask


def _preference_group_indices_for_episodes(dataset: Any, *, episode_indices: list[int]) -> np.ndarray | None:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return None
    labels: list[str] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            labels.append("")
            continue
        labels.append(str(bundle.get("merge_source_dataset_label") or bundle.get("source_dataset_label") or ""))
    nonempty_labels = sorted({label for label in labels if label})
    if not nonempty_labels:
        return None
    label_to_index = {label: index for index, label in enumerate(nonempty_labels)}
    indices = [
        label_to_index.get(labels[int(index)] if int(index) < len(labels) else "", -1) for index in episode_indices
    ]
    return np.asarray(indices, dtype=np.int64)


def _scale_optimizer_learning_rates(optimizer: Any, *, scale: float) -> dict[str, Any]:
    """Apply a run-local LR multiplier and return a reproducibility summary."""

    groups: list[dict[str, float | int]] = []
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return {"scale": float(scale), "groups": groups}
    for index, group in enumerate(param_groups):
        if not isinstance(group, dict) or "lr" not in group:
            continue
        original_lr = float(group["lr"])
        scaled_lr = original_lr * float(scale)
        group["lr"] = scaled_lr
        groups.append({"index": int(index), "original_lr": original_lr, "scaled_lr": scaled_lr})
    return {"scale": float(scale), "groups": groups}


__all__ = [
    "_initial_hidden_state",
    "_opponent_context_indices_for_episodes",
    "_parse_pair_role_selectors",
    "_parse_pair_weights",
    "_preference_group_indices_for_episodes",
    "_preference_pair_role_mask",
    "_preference_pair_weight_matrix",
    "_scale_optimizer_learning_rates",
    "_serialize_pair_role_selectors",
    "_source_opponent_policy_ids_by_episode",
]
