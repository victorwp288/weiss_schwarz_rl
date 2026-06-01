"""Detect contradictory paired-swing replay preferences."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset
from weiss_rl.training.paired_swing_conflict_filter import (
    normalize_paired_swing_action_source as _normalize_action_source,
)


@dataclass(frozen=True, slots=True)
class PairedSwingConflictConfig:
    dataset_paths: tuple[Path, ...]
    positive_action_source: str = "actions"
    negative_action_source: str = "teacher_action"
    max_examples: int = 50


def build_paired_swing_conflict_report(config: PairedSwingConflictConfig) -> dict[str, Any]:
    """Return same-observation preference conflicts across paired-swing datasets."""

    if not config.dataset_paths:
        raise ValueError("dataset_paths must contain at least one dataset")
    positive_source = _normalize_action_source(config.positive_action_source, field_name="positive_action_source")
    negative_source = _normalize_action_source(config.negative_action_source, field_name="negative_action_source")
    if positive_source == negative_source:
        raise ValueError("positive_action_source and negative_action_source must differ")

    rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    for path in config.dataset_paths:
        dataset = load_replay_trajectory_bc_dataset(path)
        start_index = len(rows)
        rows.extend(
            _iter_preference_rows(
                dataset,
                dataset_path=path,
                positive_source=positive_source,
                negative_source=negative_source,
            )
        )
        dataset_summaries.append(
            {
                "path": path.as_posix(),
                "rows": len(rows) - start_index,
                "train_rows": int(dataset.metadata.get("train_rows", 0)),
                "bundle_count": int(dataset.metadata.get("bundle_count", 0)),
            }
        )

    current_conflicts = _conflicts_for_key(rows, key_name="state_hash", max_examples=int(config.max_examples))
    history_conflicts = _conflicts_for_key(rows, key_name="history_hash", max_examples=int(config.max_examples))
    return {
        "kind": "paired_swing_conflict_report_v1",
        "dataset_paths": [path.as_posix() for path in config.dataset_paths],
        "positive_action_source": positive_source,
        "negative_action_source": negative_source,
        "preference_row_count": len(rows),
        "dataset_summaries": dataset_summaries,
        "current_state_conflict_count": len(current_conflicts),
        "history_conflict_count": len(history_conflicts),
        "current_state_conflicts": current_conflicts,
        "history_conflicts": history_conflicts,
    }


def write_paired_swing_conflict_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_preference_rows(
    dataset: ReplayTrajectoryDataset,
    *,
    dataset_path: Path,
    positive_source: str,
    negative_source: str,
) -> tuple[dict[str, Any], ...]:
    positive_actions = _dataset_actions(dataset, positive_source)
    negative_actions = _dataset_actions(dataset, negative_source)
    valid = dataset.policy_train_mask.astype(bool) & (positive_actions >= 0) & (negative_actions >= 0)
    if positive_source == "teacher_action" or negative_source == "teacher_action":
        valid &= dataset.teacher_valid.astype(bool)
    valid &= positive_actions != negative_actions

    bundles = dataset.metadata.get("selected_bundles")
    selected_bundles = bundles if isinstance(bundles, list) else []
    rows: list[dict[str, Any]] = []
    for step_index, episode_index in zip(*np.nonzero(valid), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        bundle = selected_bundles[episode] if episode < len(selected_bundles) else {}
        bundle_map = bundle if isinstance(bundle, Mapping) else {}
        rows.append(
            {
                "dataset_path": dataset_path.as_posix(),
                "step_index": step,
                "episode_index": episode,
                "source_dataset_label": str(bundle_map.get("source_dataset_label") or ""),
                "source_opponent_policy_id": str(bundle_map.get("source_opponent_policy_id") or ""),
                "source_pair_indices": _jsonable(bundle_map.get("source_pair_indices")),
                "episode_seed": _jsonable(bundle_map.get("episode_seed")),
                "positive_action": int(positive_actions[step, episode]),
                "negative_action": int(negative_actions[step, episode]),
                "state_hash": _state_hash(dataset, step_index=step, episode_index=episode),
                "history_hash": _history_hash(dataset, step_index=step, episode_index=episode),
            }
        )
    return tuple(rows)


def _conflicts_for_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_name) or "")
        if key:
            grouped[key].append(row)

    conflicts: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        positive_actions = sorted({int(row["positive_action"]) for row in group_rows})
        if len(positive_actions) <= 1:
            continue
        exact_reverse_pairs = 0
        for left_index, left in enumerate(group_rows):
            for right in group_rows[left_index + 1 :]:
                if int(left["positive_action"]) == int(right["negative_action"]) and int(
                    left["negative_action"]
                ) == int(right["positive_action"]):
                    exact_reverse_pairs += 1
        conflicts.append(
            {
                key_name: key,
                "row_count": len(group_rows),
                "positive_actions": positive_actions,
                "exact_reverse_pair_count": exact_reverse_pairs,
                "examples": [dict(row) for row in group_rows[: max(0, max_examples)]],
            }
        )
    return conflicts


def _dataset_actions(dataset: ReplayTrajectoryDataset, source: str) -> np.ndarray:
    if source == "actions":
        return np.asarray(dataset.actions, dtype=np.int64)
    if source == "teacher_action":
        return np.asarray(dataset.teacher_action, dtype=np.int64)
    raise AssertionError(f"unreachable action source: {source}")


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


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "PairedSwingConflictConfig",
    "build_paired_swing_conflict_report",
    "write_paired_swing_conflict_report",
]
