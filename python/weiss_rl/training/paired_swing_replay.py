"""In-training paired-swing replay regularizer."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, replay_trajectory_bc_batch
from weiss_rl.training.trajectory_bc_replay import TrajectoryBcReplayState

_ACTION_SOURCES = frozenset({"actions", "teacher_action"})
_CONFLICT_FILTERS = frozenset({"none", "current_state", "history"})
_COMPARE_TO_CHOICES = frozenset({"negative", "top_other"})


@dataclass(slots=True)
class PairedSwingReplayState:
    sampler: TrajectoryBcReplayState
    margin: float
    coef: float
    positive_action_source: str
    negative_action_source: str
    distinct_train_rows: int
    loss_scope: str = "row"
    compare_to: str = "negative"
    conflict_filter_summary: dict[str, Any] | None = None

    @classmethod
    def from_training_config(cls, training_config: Any, *, repo_root: Path) -> PairedSwingReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "paired_swing_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "paired_swing_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        positive_action_source = _normalize_action_source(
            getattr(structured_aux, "paired_swing_positive_action_source", "teacher_action"),
            field_name="paired_swing_positive_action_source",
        )
        negative_action_source = _normalize_action_source(
            getattr(structured_aux, "paired_swing_negative_action_source", "actions"),
            field_name="paired_swing_negative_action_source",
        )
        if positive_action_source == negative_action_source:
            raise ValueError("paired_swing_positive_action_source and paired_swing_negative_action_source must differ")
        loss_scope = str(getattr(structured_aux, "paired_swing_loss_scope", "row")).strip().lower()
        if loss_scope not in {"row", "episode_mean", "label_mean"}:
            raise ValueError("paired_swing_loss_scope must be one of: episode_mean, label_mean, row")
        compare_to = str(getattr(structured_aux, "paired_swing_compare_to", "negative")).strip().lower()
        if compare_to not in _COMPARE_TO_CHOICES:
            raise ValueError("paired_swing_compare_to must be one of: negative, top_other")
        conflict_filter = str(getattr(structured_aux, "paired_swing_conflict_filter", "none")).strip().lower()
        if conflict_filter not in _CONFLICT_FILTERS:
            raise ValueError("paired_swing_conflict_filter must be one of: current_state, history, none")
        margin = float(getattr(structured_aux, "paired_swing_margin", 0.35))
        if margin < 0.0:
            raise ValueError("paired_swing_margin must be >= 0.0")
        coef = float(getattr(structured_aux, "paired_swing_coef", 0.08))
        if coef < 0.0:
            raise ValueError("paired_swing_coef must be >= 0.0")

        sampler = TrajectoryBcReplayState.from_training_config(
            _trajectory_bc_compatible_training_config(
                structured_aux=structured_aux,
                dataset_path_text=dataset_path_text,
                every_updates=every_updates,
            ),
            repo_root=repo_root,
        )
        if sampler is None:
            return None
        conflict_filter_summary: dict[str, Any] | None = None
        if conflict_filter != "none":
            filtered_dataset, conflict_filter_summary = filter_paired_swing_conflict_rows(
                sampler.dataset,
                mode=conflict_filter,
                positive_action_source=positive_action_source,
                negative_action_source=negative_action_source,
            )
            sampler.dataset = filtered_dataset
            sampler.order = sampler.rng.permutation(filtered_dataset.episode_count)
            sampler.cursor = 0
            sampler.focus_cursor = 0
            sampler.nonfocus_cursor = 0
        distinct_train_rows = paired_swing_distinct_train_row_count(
            sampler.dataset,
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
        )
        if distinct_train_rows <= 0:
            raise ValueError(
                "paired-swing dataset has no trainable rows where positive and negative actions differ: "
                f"{dataset_path_text}"
            )
        return cls(
            sampler=sampler,
            margin=margin,
            coef=coef,
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
            distinct_train_rows=distinct_train_rows,
            loss_scope=loss_scope,
            compare_to=compare_to,
            conflict_filter_summary=conflict_filter_summary,
        )


def maybe_run_paired_swing_replay(
    *,
    state: PairedSwingReplayState | None,
    learner: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured paired-swing auxiliary steps after an RL update."""

    if state is None:
        return
    sampler = state.sampler
    if int(update_count) <= 0 or int(update_count) % int(sampler.every_updates) != 0:
        return
    updater = getattr(learner, "paired_swing_update", None)
    if not callable(updater):
        raise ValueError("learner does not support paired_swing_update")

    aux_metrics: dict[str, float] = {}
    total_batch_episodes = 0
    total_focus_episodes = 0
    total_nonfocus_episodes = 0
    total_context_episodes = 0
    total_focus_group_counts = {group.name: 0 for group in sampler.focus_groups}
    for _ in range(int(sampler.aux_updates)):
        indices = sampler.next_episode_indices()
        total_batch_episodes += len(indices)
        total_focus_episodes += int(sampler.last_focus_episode_count)
        total_nonfocus_episodes += int(sampler.last_nonfocus_episode_count)
        for group in sampler.focus_groups:
            total_focus_group_counts[group.name] = total_focus_group_counts.get(group.name, 0) + int(
                group.last_episode_count
            )
        opponent_context_indices = _opponent_context_indices_for_episodes(
            learner.model,
            sampler.dataset,
            episode_indices=indices,
        )
        if opponent_context_indices is not None:
            total_context_episodes += int(np.count_nonzero(opponent_context_indices))
        hidden = _initial_hidden_state(
            learner.model,
            batch_size=len(indices),
            device=device,
            opponent_context_indices=opponent_context_indices,
        )
        batch = replay_trajectory_bc_batch(
            sampler.dataset,
            episode_indices=indices,
            initial_hidden_state=hidden,
            opponent_context_indices=opponent_context_indices,
        )
        aux_metrics = updater(
            batch,
            margin=float(state.margin),
            coef=float(state.coef),
            positive_action_source=state.positive_action_source,
            negative_action_source=state.negative_action_source,
            loss_scope=state.loss_scope,
            compare_to=state.compare_to,
        )

    latest_metrics["paired_swing_replay_aux_updates"] = float(sampler.aux_updates)
    latest_metrics["paired_swing_replay_batch_episodes"] = float(total_batch_episodes)
    latest_metrics["paired_swing_replay_dataset_train_rows"] = float(sampler.dataset.metadata["train_rows"])
    latest_metrics["paired_swing_replay_dataset_distinct_train_rows"] = float(state.distinct_train_rows)
    latest_metrics["paired_swing_replay_focus_fraction"] = float(sampler.focus_fraction)
    latest_metrics["paired_swing_replay_focus_batch_episodes"] = float(total_focus_episodes)
    latest_metrics["paired_swing_replay_nonfocus_batch_episodes"] = float(total_nonfocus_episodes)
    latest_metrics["paired_swing_replay_margin"] = float(state.margin)
    latest_metrics["paired_swing_replay_coef"] = float(state.coef)
    latest_metrics["paired_swing_replay_loss_scope_episode_mean"] = 1.0 if state.loss_scope == "episode_mean" else 0.0
    latest_metrics["paired_swing_replay_loss_scope_label_mean"] = 1.0 if state.loss_scope == "label_mean" else 0.0
    if state.compare_to == "top_other":
        latest_metrics["paired_swing_replay_compare_to_top_other"] = 1.0
    latest_metrics["paired_swing_replay_opponent_context_episodes"] = float(total_context_episodes)
    if state.conflict_filter_summary is not None:
        summary = state.conflict_filter_summary
        latest_metrics["paired_swing_replay_conflict_filter_active"] = 1.0
        latest_metrics["paired_swing_replay_conflict_filter_dropped_rows"] = float(summary.get("dropped_train_rows", 0))
        latest_metrics["paired_swing_replay_conflict_filter_kept_rows"] = float(summary.get("kept_train_rows", 0))
        latest_metrics["paired_swing_replay_conflict_filter_conflict_keys"] = float(
            summary.get("conflict_key_count", 0)
        )
    if state.positive_action_source == "teacher_action":
        latest_metrics["paired_swing_replay_positive_source_teacher"] = 1.0
    if state.negative_action_source == "teacher_action":
        latest_metrics["paired_swing_replay_negative_source_teacher"] = 1.0
    if sampler.focus_groups:
        latest_metrics["paired_swing_replay_focus_group_count"] = float(len(sampler.focus_groups))
        for group in sampler.focus_groups:
            key = _metric_key_fragment(group.name)
            latest_metrics[f"paired_swing_replay_focus_group_{key}_batch_episodes"] = float(
                total_focus_group_counts.get(group.name, 0)
            )
    for key, value in aux_metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            latest_metrics[f"paired_swing_replay_{key}"] = float(value)


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
    if normalized_mode not in _CONFLICT_FILTERS - {"none"}:
        raise ValueError("paired-swing conflict filter mode must be current_state or history")
    positive_source = _normalize_action_source(positive_action_source, field_name="positive_action_source")
    negative_source = _normalize_action_source(negative_action_source, field_name="negative_action_source")
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
        row = {
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


def _trajectory_bc_compatible_training_config(
    *,
    structured_aux: Any,
    dataset_path_text: str,
    every_updates: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            trajectory_bc_dataset_path=dataset_path_text,
            trajectory_bc_every_updates=every_updates,
            trajectory_bc_aux_updates=int(getattr(structured_aux, "paired_swing_aux_updates", 1)),
            trajectory_bc_batch_episodes=int(getattr(structured_aux, "paired_swing_batch_episodes", 8)),
            trajectory_bc_seed=int(getattr(structured_aux, "paired_swing_seed", 20260519)),
            trajectory_bc_focus_source_labels=tuple(getattr(structured_aux, "paired_swing_focus_source_labels", ())),
            trajectory_bc_focus_fraction=float(getattr(structured_aux, "paired_swing_focus_fraction", 0.0)),
            trajectory_bc_focus_groups=tuple(getattr(structured_aux, "paired_swing_focus_groups", ())),
        )
    )


def _dataset_actions(dataset: ReplayTrajectoryDataset, source: str) -> np.ndarray:
    normalized = _normalize_action_source(source, field_name="action_source")
    if normalized == "actions":
        return np.asarray(dataset.actions, dtype=np.int64)
    if normalized == "teacher_action":
        return np.asarray(dataset.teacher_action, dtype=np.int64)
    raise AssertionError(f"unreachable action source: {normalized}")


def _exact_reverse_pair_count(rows: list[Mapping[str, int | str]]) -> int:
    count = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if int(left["positive_action"]) == int(right["negative_action"]) and int(left["negative_action"]) == int(
                right["positive_action"]
            ):
                count += 1
    return count


def _normalize_action_source(value: object, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _ACTION_SOURCES:
        raise ValueError(f"{field_name} must be one of: actions, teacher_action")
    return normalized


def _opponent_context_indices_for_episodes(
    model: Any,
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: list[int],
) -> np.ndarray | None:
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return None
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    if not opponent_ids:
        return None
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    indices = model.opponent_context_indices_for_policy_ids(selected_policy_ids)
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def _source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, dict) else None
        ids.append(str(raw_id or "").strip())
    return ids


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


def _initial_hidden_state(
    model: Any,
    *,
    batch_size: int,
    device: torch.device,
    opponent_context_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
    kwargs: dict[str, Any] = {"device": device}
    if opponent_context_indices is not None:
        kwargs["opponent_context_indices"] = opponent_context_indices
    try:
        hidden = model.initial_seat_hidden(int(batch_size), **kwargs)
    except TypeError:
        hidden = model.initial_seat_hidden(int(batch_size), device=device)
    return hidden.detach().cpu().numpy()


def _metric_key_fragment(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "group"


__all__ = [
    "PairedSwingReplayState",
    "filter_paired_swing_conflict_rows",
    "maybe_run_paired_swing_replay",
    "paired_swing_distinct_train_row_count",
]
