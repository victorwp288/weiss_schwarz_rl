"""Paired-swing dataset filtering and trainable-row accounting."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset

PAIRED_SWING_ACTION_SOURCES = frozenset({"actions", "teacher_action"})
PAIRED_SWING_CONFLICT_FILTERS = frozenset({"none", "current_state", "history"})


def paired_swing_distinct_train_row_count(
    dataset: ReplayTrajectoryDataset,
    *,
    positive_action_source: str = "teacher_action",
    negative_action_source: str = "actions",
) -> int:
    positive_actions = _dataset_actions(dataset, positive_action_source)
    negative_actions = _dataset_actions(dataset, negative_action_source)
    valid = (
        dataset.policy_train_mask.astype(bool)
        & (positive_actions >= 0)
        & (negative_actions >= 0)
        & (positive_actions != negative_actions)
    )
    if positive_action_source == "teacher_action" or negative_action_source == "teacher_action":
        valid &= dataset.teacher_valid.astype(bool)
    if not bool(np.any(valid)):
        return 0
    flat_valid = valid.reshape(-1)
    flat_positive = positive_actions.reshape(-1)
    flat_negative = negative_actions.reshape(-1)
    legal_offsets = np.asarray(dataset.legal_offsets, dtype=np.int64)
    legal_ids = np.asarray(dataset.legal_ids, dtype=np.int64)
    count = 0
    for row_index in np.flatnonzero(flat_valid).astype(np.int64).tolist():
        start = int(legal_offsets[row_index])
        stop = int(legal_offsets[row_index + 1])
        row_ids = legal_ids[start:stop]
        if np.any(row_ids == int(flat_positive[row_index])) and np.any(row_ids == int(flat_negative[row_index])):
            count += 1
    return int(count)


def filter_paired_swing_conflict_rows(
    dataset: ReplayTrajectoryDataset,
    *,
    mode: str,
    positive_action_source: str = "teacher_action",
    negative_action_source: str = "actions",
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Mask paired-swing rows whose state/history asks for contradictory positives."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in PAIRED_SWING_CONFLICT_FILTERS - {"none"}:
        raise ValueError("paired-swing conflict filter mode must be current_state or history")
    positive_source = normalize_paired_swing_action_source(
        positive_action_source,
        field_name="positive_action_source",
    )
    negative_source = normalize_paired_swing_action_source(
        negative_action_source,
        field_name="negative_action_source",
    )
    positive_actions = _dataset_actions(dataset, positive_source)
    negative_actions = _dataset_actions(dataset, negative_source)
    valid = (
        dataset.policy_train_mask.astype(bool)
        & (positive_actions >= 0)
        & (negative_actions >= 0)
        & (positive_actions != negative_actions)
    )
    if positive_source == "teacher_action" or negative_source == "teacher_action":
        valid &= dataset.teacher_valid.astype(bool)

    preference_rows: list[dict[str, int | str]] = []
    grouped: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for step_index, episode_index in zip(*np.nonzero(valid), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        key = (
            _state_hash(dataset, step_index=step, episode_index=episode)
            if normalized_mode == "current_state"
            else _history_hash(dataset, step_index=step, episode_index=episode)
        )
        row: dict[str, int | str] = {
            "key": key,
            "step_index": step,
            "episode_index": episode,
            "positive_action": int(positive_actions[step, episode]),
            "negative_action": int(negative_actions[step, episode]),
        }
        preference_rows.append(row)
        grouped[key].append(row)

    conflict_keys: set[str] = set()
    exact_reverse_pair_count = 0
    for key, rows in grouped.items():
        positive_set = {int(row["positive_action"]) for row in rows}
        reverse_pairs = _exact_reverse_pair_count(rows)
        exact_reverse_pair_count += reverse_pairs
        if len(positive_set) > 1 or reverse_pairs > 0:
            conflict_keys.add(key)

    filtered_mask = dataset.policy_train_mask.astype(bool).copy()
    dropped = 0
    for row in preference_rows:
        if str(row["key"]) not in conflict_keys:
            continue
        step = int(row["step_index"])
        episode = int(row["episode_index"])
        if bool(filtered_mask[step, episode]):
            filtered_mask[step, episode] = False
            dropped += 1

    before_train_rows = int(np.count_nonzero(dataset.policy_train_mask))
    kept_train_rows = int(np.count_nonzero(filtered_mask))
    summary = {
        "kind": "paired_swing_conflict_filter_v1",
        "mode": normalized_mode,
        "positive_action_source": positive_source,
        "negative_action_source": negative_source,
        "preference_row_count": len(preference_rows),
        "conflict_key_count": len(conflict_keys),
        "exact_reverse_pair_count": int(exact_reverse_pair_count),
        "before_train_rows": before_train_rows,
        "dropped_train_rows": int(dropped),
        "kept_train_rows": kept_train_rows,
    }
    metadata = dict(dataset.metadata)
    metadata["train_rows"] = kept_train_rows
    metadata["paired_swing_conflict_filter"] = summary
    return replace(dataset, policy_train_mask=filtered_mask, metadata=metadata), summary


def normalize_paired_swing_action_source(value: object, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in PAIRED_SWING_ACTION_SOURCES:
        raise ValueError(f"{field_name} must be one of: actions, teacher_action")
    return normalized


def _dataset_actions(dataset: ReplayTrajectoryDataset, source: str) -> np.ndarray:
    normalized = normalize_paired_swing_action_source(source, field_name="action_source")
    if normalized == "actions":
        return np.asarray(dataset.actions, dtype=np.int64)
    if normalized == "teacher_action":
        return np.asarray(dataset.teacher_action, dtype=np.int64)
    raise AssertionError(f"unreachable action source: {normalized}")


def _exact_reverse_pair_count(rows: Sequence[Mapping[str, int | str]]) -> int:
    count = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if int(left["positive_action"]) == int(right["negative_action"]) and int(left["negative_action"]) == int(
                right["positive_action"]
            ):
                count += 1
    return count


def _state_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    row_index = int(step_index) * int(dataset.episode_count) + int(episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    return _hash_arrays(
        np.asarray(dataset.obs[step_index, episode_index]),
        np.asarray(
            [dataset.actor[step_index, episode_index], dataset.to_play_seat[step_index, episode_index]],
            dtype=np.int64,
        ),
        np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32),
    )


def _history_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    stop = int(step_index) + 1
    return _hash_arrays(
        np.asarray(dataset.obs[:stop, episode_index]),
        np.asarray(dataset.actor[:stop, episode_index], dtype=np.int64),
        np.asarray(dataset.to_play_seat[:stop, episode_index], dtype=np.int64),
        np.asarray(dataset.reset_before_step[:stop, episode_index], dtype=np.bool_),
    )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


__all__ = [
    "PAIRED_SWING_ACTION_SOURCES",
    "PAIRED_SWING_CONFLICT_FILTERS",
    "filter_paired_swing_conflict_rows",
    "normalize_paired_swing_action_source",
    "paired_swing_distinct_train_row_count",
]
