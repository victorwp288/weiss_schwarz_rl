"""Trajectory provenance records for replay inspection reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.observation_layout import ObservationPlayerBlock, ObservationSlice
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.models.observations.observation_contract import header_field_index
from weiss_rl.replay.bundles import ReplayStep
from weiss_rl.replay.inspection_step_diffs import action_descriptor
from weiss_rl.replay.inspection_summaries import TRACKED_LEGAL_FAMILIES, counter_items
from weiss_rl.replay.rerun_validation import load_observation_layout, pass_action_id_from_spec_bundle


def build_trajectory_record(
    *,
    step_index: int,
    expected_step: ReplayStep,
    batch: DecisionBoundaryBatch,
    raw_legal_ids: np.ndarray,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    obs = np.asarray(batch.obs[0])
    layout = load_observation_layout(spec_bundle)
    recorded_action = action_descriptor(int(expected_step.action), action_catalog=action_catalog)
    legal_family_counts = _legal_family_counts(raw_legal_ids=raw_legal_ids, action_catalog=action_catalog)
    pass_action_id = pass_action_id_from_spec_bundle(spec_bundle)
    payload: dict[str, Any] = {
        "step_index": int(step_index),
        "decision_id": int(expected_step.decision_id),
        "actor": int(expected_step.actor),
        "recorded_action": int(expected_step.action),
        "recorded_action_family": str(recorded_action.get("family", "unknown")),
        "raw_legal_action_count": int(np.asarray(raw_legal_ids).shape[0]),
        "has_nonpass_legal": bool(np.any(np.asarray(raw_legal_ids, dtype=np.int64) != pass_action_id)),
        "legal_family_counts": counter_items(legal_family_counts, key_names=("family",)),
    }
    for family in TRACKED_LEGAL_FAMILIES:
        payload[f"has_legal_{family}"] = int(legal_family_counts.get(family, 0)) > 0
    if layout is not None:
        for field_name in ("phase", "decision_kind", "active_player", "decision_player"):
            field_index = header_field_index(layout, field_name)
            if field_index is not None and field_index < obs.shape[0]:
                payload[field_name] = _safe_int(obs[field_index])
        if layout.player_blocks:
            payload.update(
                _player_trajectory_fields(
                    obs=obs,
                    block=layout.player_blocks[0],
                    prefix="self",
                    action_catalog=action_catalog,
                    spec_bundle=spec_bundle,
                )
            )
        if len(layout.player_blocks) > 1:
            payload.update(
                _player_trajectory_fields(
                    obs=obs,
                    block=layout.player_blocks[1],
                    prefix="opponent",
                    action_catalog=action_catalog,
                    spec_bundle=spec_bundle,
                )
            )
    return payload


def _player_trajectory_fields(
    *,
    obs: np.ndarray,
    block: ObservationPlayerBlock,
    prefix: str,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in (
        "level_count",
        "clock_count",
        "deck_count",
        "hand_count",
        "stock_count",
        "waiting_room_count",
        "memory_count",
        "climax_count",
    ):
        value = _player_scalar(obs=obs, block=block, slice_name=name)
        if value is not None:
            payload[f"{prefix}_{name}"] = value
    stage_slice = _player_slice(block, "stage")
    if stage_slice is not None:
        payload[f"{prefix}_stage_occupied_count"] = _stage_occupied_count(
            obs=obs,
            stage_start=stage_slice.start,
            stage_length=stage_slice.length,
            action_catalog=action_catalog,
            spec_bundle=spec_bundle,
        )
    return payload


def _player_scalar(*, obs: np.ndarray, block: ObservationPlayerBlock, slice_name: str) -> int | None:
    observation_slice = _player_slice(block, slice_name)
    if observation_slice is None or observation_slice.start >= obs.shape[0]:
        return None
    return _safe_int(obs[observation_slice.start])


def _player_slice(block: ObservationPlayerBlock, slice_name: str) -> ObservationSlice | None:
    return next((current for current in block.slices if current.name == slice_name), None)


def _stage_occupied_count(
    *,
    obs: np.ndarray,
    stage_start: int,
    stage_length: int,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> int:
    stage_slots = int(action_catalog.max_stage) if action_catalog is not None else 5
    if stage_slots <= 0:
        return 0
    slot_width = max(int(stage_length) // stage_slots, 1)
    observation = spec_bundle.get("observation") if isinstance(spec_bundle, Mapping) else None
    sentinel_empty = 0
    sentinel_hidden = -1
    if isinstance(observation, Mapping):
        sentinel_empty = int(observation.get("sentinel_empty_card", sentinel_empty))
        sentinel_hidden = int(observation.get("sentinel_hidden", sentinel_hidden))
    occupied = 0
    for slot_index in range(stage_slots):
        index = int(stage_start) + slot_index * slot_width
        if index >= obs.shape[0] or index >= int(stage_start) + int(stage_length):
            break
        card_value = _safe_int(obs[index])
        if card_value not in (sentinel_empty, sentinel_hidden):
            occupied += 1
    return occupied


def _legal_family_counts(*, raw_legal_ids: np.ndarray, action_catalog: ActionCatalog | None) -> Counter[str]:
    counter: Counter[str] = Counter()
    if action_catalog is None:
        return counter
    for action_id in np.asarray(raw_legal_ids, dtype=np.int64).tolist():
        try:
            family = action_catalog.decode(int(action_id)).family
        except ValueError:
            family = "unknown"
        counter[str(family)] += 1
    return counter


def _safe_int(value: Any) -> int:
    return int(np.asarray(value).reshape(()).item())


__all__ = [
    "build_trajectory_record",
]
