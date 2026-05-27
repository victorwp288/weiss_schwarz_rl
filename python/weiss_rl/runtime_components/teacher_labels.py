"""Teacher-label helpers for runtime collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

TeacherLabelArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
PUBLIC_TEACHER_DECISION_KINDS = frozenset({1, 2, 3, 4, 5, 6, 7, 8})
SUPPORTED_TEACHER_LABEL_PROFILES = frozenset({"base", "aggressive", "control"})


def selected_teacher_label_profile(
    profiles: tuple[str, ...] | list[str] | None,
    *,
    profile_mode: str,
    update_count: int,
    end_updates: int,
) -> str:
    """Select the heuristic profile used for exact teacher-action labels."""

    normalized: list[str] = []
    for raw_profile in profiles or ():
        profile = str(raw_profile).strip().lower()
        if profile and profile not in normalized:
            normalized.append(profile)
    if not normalized:
        normalized = ["base"]
    invalid = sorted(set(normalized) - SUPPORTED_TEACHER_LABEL_PROFILES)
    if invalid:
        raise ValueError("teacher label profiles contain unsupported profiles: " + ", ".join(invalid))

    if int(end_updates) >= 0 and int(update_count) > int(end_updates):
        return normalized[0]
    if str(profile_mode).strip().lower() == "cycle" and len(normalized) > 1:
        return normalized[int(update_count) % len(normalized)]
    return normalized[0]


def teacher_guidance_active_for_collection(
    *,
    enabled: bool,
    teacher_aux_mode: str,
    warmstart_updates: int,
    current_learner_update: int,
) -> bool:
    """Return whether teacher labels should be collected for the current update."""

    if not bool(enabled):
        return False
    mode = str(teacher_aux_mode).strip().lower()
    if mode == "off":
        return False
    if mode != "warmstart_only":
        return True
    warmstart_limit = max(0, int(warmstart_updates))
    if warmstart_limit <= 0:
        return False
    return int(current_learner_update) < warmstart_limit


def teacher_label_arrays(num_rows: int) -> TeacherLabelArrays:
    """Create default sentinel teacher-label arrays for one collection step."""

    shape = (int(num_rows),)
    return (
        np.full(shape, -1, dtype=np.int32),
        np.full(shape, -1, dtype=np.int32),
        np.full(shape, -1, dtype=np.int32),
        np.full(shape, -1, dtype=np.int32),
        np.full(shape, -1, dtype=np.int32),
        np.zeros(shape, dtype=np.bool_),
    )


def teacher_labels_from_actions(
    *,
    row_indices: np.ndarray,
    chosen_actions: np.ndarray,
    num_rows: int,
    guidance_active: bool,
    action_catalog: Any | None,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> TeacherLabelArrays:
    """Decode chosen action ids into structured teacher-label arrays."""

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        teacher_label_arrays(num_rows)
    )
    if not bool(guidance_active) or action_catalog is None:
        return teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid

    for row_index, action_id in zip(
        np.asarray(row_indices, dtype=np.int64).tolist(),
        np.asarray(chosen_actions, dtype=np.int64).tolist(),
        strict=True,
    ):
        decoded = action_catalog.decode(int(action_id))
        decoded_family = str(decoded.family)
        family_id = family_index.get(decoded_family)
        if family_id is None:
            continue
        row = int(row_index)
        teacher_valid[row] = True
        teacher_family[row] = int(family_id)
        teacher_action[row] = int(action_id)
        if decoded_family == "main_play_character" and decoded.stage_slot is not None:
            teacher_slot[row] = int(decoded.stage_slot)
        elif decoded_family == "main_move" and decoded.to_slot is not None:
            teacher_slot[row] = int(decoded.to_slot)
            if decoded.from_slot is not None:
                teacher_move_source[row] = int(decoded.from_slot)
        elif decoded_family == "attack":
            if decoded.slot is not None:
                teacher_slot[row] = int(decoded.slot)
            if decoded.attack_type is not None:
                attack_type_id = attack_type_index.get(str(decoded.attack_type))
                if attack_type_id is not None:
                    teacher_attack_type[row] = int(attack_type_id)
    return teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid


def teacher_labels_from_ids(
    *,
    focal_rows: np.ndarray,
    decision_kind: np.ndarray,
    obs_step: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
    counters: dict[str, int] | None,
    guidance_active: bool,
    teacher_policy: Any | None,
    action_catalog: Any | None,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    select_actions_from_ids: Callable[..., np.ndarray],
) -> TeacherLabelArrays:
    num_rows = int(decision_kind.shape[0])
    labels = teacher_label_arrays(num_rows)
    if not bool(guidance_active) or teacher_policy is None:
        return labels
    decision_kind_array = np.asarray(decision_kind, dtype=np.int32)
    teacher_rows = np.flatnonzero(
        np.asarray(focal_rows, dtype=np.bool_) & np.isin(decision_kind_array, tuple(PUBLIC_TEACHER_DECISION_KINDS))
    )
    if teacher_rows.size == 0:
        return labels
    if counters is not None:
        counters["teacher_tactical_row_count"] += int(teacher_rows.size)
    chosen_actions = select_actions_from_ids(
        actor=None,
        heuristic_policy=teacher_policy,
        row_indices=teacher_rows,
        obs_step=obs_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        counters=counters,
    )
    return teacher_labels_from_actions(
        row_indices=teacher_rows,
        chosen_actions=chosen_actions,
        num_rows=num_rows,
        guidance_active=guidance_active,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )


def teacher_labels_from_mask(
    *,
    focal_rows: np.ndarray,
    decision_kind: np.ndarray,
    obs_step: np.ndarray,
    legal_mask: np.ndarray,
    counters: dict[str, int] | None,
    guidance_active: bool,
    teacher_policy: Any | None,
    action_catalog: Any | None,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    select_actions_from_mask: Callable[..., np.ndarray],
) -> TeacherLabelArrays:
    num_rows = int(decision_kind.shape[0])
    labels = teacher_label_arrays(num_rows)
    if not bool(guidance_active) or teacher_policy is None:
        return labels
    decision_kind_array = np.asarray(decision_kind, dtype=np.int32)
    teacher_rows = np.flatnonzero(
        np.asarray(focal_rows, dtype=np.bool_) & np.isin(decision_kind_array, tuple(PUBLIC_TEACHER_DECISION_KINDS))
    )
    if teacher_rows.size == 0:
        return labels
    if counters is not None:
        counters["teacher_tactical_row_count"] += int(teacher_rows.size)
    chosen_actions = select_actions_from_mask(
        actor=None,
        heuristic_policy=teacher_policy,
        row_indices=teacher_rows,
        obs_step=obs_step,
        legal_mask=legal_mask,
        counters=counters,
    )
    return teacher_labels_from_actions(
        row_indices=teacher_rows,
        chosen_actions=chosen_actions,
        num_rows=num_rows,
        guidance_active=guidance_active,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
