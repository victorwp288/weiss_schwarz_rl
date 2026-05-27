"""Prototype-key coverage diagnostics for paired outcome preference replay."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config import load_stack_config
from weiss_rl.experiments.paired_outcome_preference_decisions import (
    _first_optional_int,
    _history_hash,
    _selected_bundles,
    _state_hash,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset

_VALID_KEY_MODES = {"current", "current_history", "current_history_opponent"}
_VALID_OPPONENT_KEY_MODES = {"raw_policy_id", "context_index"}


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSurfacePrototypeConfig:
    prototype_dataset_path: Path
    probe_dataset_paths: tuple[Path, ...]
    probe_labels: tuple[str, ...] = ()
    stack_config_path: Path | None = None
    opponent_context_policy_ids: tuple[str, ...] = ()
    key_mode: str = "current_history_opponent"
    opponent_key_mode: str = "raw_policy_id"
    max_examples: int = 20


def build_paired_outcome_preference_surface_prototype_report(
    config: PairedOutcomePreferenceSurfacePrototypeConfig,
) -> dict[str, Any]:
    """Return exact prototype-key coverage/leakage diagnostics for replay datasets."""

    key_mode = str(config.key_mode)
    if key_mode not in _VALID_KEY_MODES:
        raise ValueError(f"key_mode must be one of {sorted(_VALID_KEY_MODES)}, got {key_mode!r}")
    opponent_key_mode = str(config.opponent_key_mode)
    if opponent_key_mode not in _VALID_OPPONENT_KEY_MODES:
        raise ValueError(
            f"opponent_key_mode must be one of {sorted(_VALID_OPPONENT_KEY_MODES)}, got {opponent_key_mode!r}"
        )
    context_policy_ids = _opponent_context_policy_ids(config)
    if key_mode == "current_history_opponent" and opponent_key_mode == "context_index" and not context_policy_ids:
        raise ValueError("opponent_key_mode='context_index' requires opponent_context_policy_ids or stack_config_path")
    context_index_by_policy_id = {
        policy_id: index for index, policy_id in enumerate(context_policy_ids, start=1) if policy_id
    }

    prototype_dataset = load_replay_trajectory_bc_dataset(config.prototype_dataset_path)
    prototype_rows = _train_rows(
        prototype_dataset,
        context_index_by_policy_id=context_index_by_policy_id,
    )
    prototype_groups = _group_rows_by_key(prototype_rows, key_mode=key_mode, opponent_key_mode=opponent_key_mode)
    prototype_key_set = set(prototype_groups)
    prototype_source_keys = {
        _source_key(row)
        for row in prototype_rows
        if row["source_opponent_policy_id"] or row["source_pair_index"] is not None
    }
    max_examples = max(0, int(config.max_examples))

    probe_reports: list[dict[str, Any]] = []
    for probe_index, probe_path in enumerate(config.probe_dataset_paths):
        label = (
            str(config.probe_labels[probe_index])
            if probe_index < len(config.probe_labels) and str(config.probe_labels[probe_index]).strip()
            else Path(probe_path).stem
        )
        probe_reports.append(
            _probe_report(
                label=label,
                dataset_path=Path(probe_path),
                key_mode=key_mode,
                opponent_key_mode=opponent_key_mode,
                context_index_by_policy_id=context_index_by_policy_id,
                prototype_key_set=prototype_key_set,
                prototype_source_keys=prototype_source_keys,
                max_examples=max_examples,
            )
        )

    return {
        "kind": "paired_outcome_preference_surface_prototype_report_v1",
        "prototype_dataset_path": config.prototype_dataset_path.as_posix(),
        "stack_config_path": None if config.stack_config_path is None else config.stack_config_path.as_posix(),
        "key_mode": key_mode,
        "opponent_key_mode": opponent_key_mode,
        "key_definition": _key_definition(key_mode, opponent_key_mode=opponent_key_mode),
        "model_input_fields": [
            "obs/current_state",
            "actor",
            "to_play_seat",
            "legal_ids",
            "legal_action_meta",
            "history/recurrent_state when key_mode includes history",
            "source_opponent_policy_id when key_mode includes opponent and opponent_key_mode is raw_policy_id",
            "opponent_context_index when key_mode includes opponent and opponent_key_mode is context_index",
        ],
        "diagnostic_only_fields": [
            "preference_pair_id",
            "preference_role",
            "preference_role_label",
            "source_pair_index",
            "episode_seed",
            "swap_index",
            "source_dataset_label",
            "merge_source_dataset_label",
        ],
        "opponent_context_policy_ids": list(context_policy_ids),
        "prototype": _dataset_summary(
            label="prototype",
            dataset_path=config.prototype_dataset_path,
            rows=prototype_rows,
            groups=prototype_groups,
            max_examples=max_examples,
        ),
        "probes": probe_reports,
    }


def write_paired_outcome_preference_surface_prototype_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _probe_report(
    *,
    label: str,
    dataset_path: Path,
    key_mode: str,
    opponent_key_mode: str,
    context_index_by_policy_id: Mapping[str, int],
    prototype_key_set: set[tuple[str, ...]],
    prototype_source_keys: set[tuple[str, int | None]],
    max_examples: int,
) -> dict[str, Any]:
    dataset = load_replay_trajectory_bc_dataset(dataset_path)
    rows = _train_rows(dataset, context_index_by_policy_id=context_index_by_policy_id)
    groups = _group_rows_by_key(rows, key_mode=key_mode, opponent_key_mode=opponent_key_mode)
    matched_rows = [
        row
        for row in rows
        if _key_for_row(row, key_mode=key_mode, opponent_key_mode=opponent_key_mode) in prototype_key_set
    ]
    matched_groups = _group_rows_by_key(matched_rows, key_mode=key_mode, opponent_key_mode=opponent_key_mode)
    unexpected_rows = [row for row in matched_rows if _source_key(row) not in prototype_source_keys]
    total_rows = len(rows)
    return {
        "label": label,
        "dataset_path": dataset_path.as_posix(),
        "train_rows": total_rows,
        "unique_key_count": len(groups),
        "matched_train_rows": len(matched_rows),
        "matched_rate": 0.0 if total_rows <= 0 else len(matched_rows) / float(total_rows),
        "matched_unique_key_count": len(matched_groups),
        "unexpected_matched_rows": len(unexpected_rows),
        "unexpected_matched_rate": 0.0 if len(matched_rows) <= 0 else len(unexpected_rows) / float(len(matched_rows)),
        "matched_summary": _row_collection_summary(matched_rows),
        "unexpected_matched_summary": _row_collection_summary(unexpected_rows),
        "conflicting_matched_key_count": _conflicting_key_count(matched_groups),
        "ambiguous_matched_keys": _ambiguous_key_rows(matched_groups, max_examples=max_examples),
        "unexpected_examples": [_row_example(row) for row in unexpected_rows[:max_examples]],
        "matched_examples": [_row_example(row) for row in matched_rows[:max_examples]],
    }


def _dataset_summary(
    *,
    label: str,
    dataset_path: Path,
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    max_examples: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "dataset_path": dataset_path.as_posix(),
        "train_rows": len(rows),
        "unique_key_count": len(groups),
        "conflicting_key_count": _conflicting_key_count(groups),
        "summary": _row_collection_summary(rows),
        "ambiguous_keys": _ambiguous_key_rows(groups, max_examples=max_examples),
        "examples": [_row_example(row) for row in rows[:max_examples]],
    }


def _train_rows(
    dataset: ReplayTrajectoryDataset,
    *,
    context_index_by_policy_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    bundles = _selected_bundles(dataset)
    rows: list[dict[str, Any]] = []
    for step_index, episode_index in zip(*np.nonzero(dataset.policy_train_mask.astype(bool)), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        bundle = bundles[episode]
        opponent_policy_id = str(bundle.get("source_opponent_policy_id") or "")
        rows.append(
            {
                "step_index": step,
                "episode_index": episode,
                "action": int(dataset.actions[step, episode]),
                "current_state_hash": _state_hash(dataset, step_index=step, episode_index=episode),
                "history_hash": _history_hash(dataset, step_index=step, episode_index=episode),
                "legal_candidate_hash": _legal_candidate_hash(dataset, step_index=step, episode_index=episode),
                "source_opponent_policy_id": opponent_policy_id,
                "opponent_context_index": _context_index_for_policy_id(
                    opponent_policy_id,
                    context_index_by_policy_id,
                ),
                "source_pair_index": _first_optional_int(bundle, ("source_pair_index", "pair_index")),
                "preference_pair_id": _optional_int(bundle.get("preference_pair_id")),
                "preference_role": _optional_int(bundle.get("preference_role")),
                "preference_role_label": str(bundle.get("preference_role_label") or ""),
                "source_dataset_label": _source_dataset_label(bundle),
            }
        )
    return rows


def _group_rows_by_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_mode: str,
    opponent_key_mode: str,
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_key_for_row(row, key_mode=key_mode, opponent_key_mode=opponent_key_mode)].append(row)
    return dict(groups)


def _key_for_row(row: Mapping[str, Any], *, key_mode: str, opponent_key_mode: str) -> tuple[str, ...]:
    current = str(row.get("current_state_hash") or "")
    legal = str(row.get("legal_candidate_hash") or "")
    if key_mode == "current":
        return ("current", current, legal)
    history = str(row.get("history_hash") or "")
    if key_mode == "current_history":
        return ("current_history", current, history, legal)
    if opponent_key_mode == "context_index":
        opponent = f"context_index:{int(row.get('opponent_context_index') or 0)}"
    else:
        opponent = str(row.get("source_opponent_policy_id") or "")
    if key_mode == "current_history_opponent":
        return ("current_history_opponent", current, history, legal, opponent)
    raise ValueError(f"unsupported key_mode {key_mode!r}")


def _legal_candidate_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    row_index = int(step_index) * int(dataset.episode_count) + int(episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    digest = hashlib.sha256()
    for array in (
        np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32),
        np.asarray(dataset.legal_action_meta[start:stop]),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _key_definition(key_mode: str, *, opponent_key_mode: str) -> dict[str, Any]:
    fields = ["current_state_hash", "legal_candidate_hash"]
    if key_mode in {"current_history", "current_history_opponent"}:
        fields.insert(1, "history_hash")
    if key_mode == "current_history_opponent":
        if opponent_key_mode == "context_index":
            fields.append("opponent_context_index")
        else:
            fields.append("source_opponent_policy_id")
    return {
        "mode": key_mode,
        "key_fields": fields,
        "current_state_hash": ["obs[step, episode]", "actor", "to_play_seat", "legal_ids"],
        "history_hash": ["obs[:step+1, episode]", "actor[:step+1]", "to_play_seat[:step+1]", "reset_before_step"],
        "legal_candidate_hash": ["legal_ids", "legal_action_meta"],
    }


def _row_collection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "opponents": _counter_rows(rows, "source_opponent_policy_id"),
        "opponent_context_indices": _counter_rows(rows, "opponent_context_index"),
        "source_pair_indices": _counter_rows(rows, "source_pair_index"),
        "source_dataset_labels": _counter_rows(rows, "source_dataset_label"),
        "preference_role_labels": _counter_rows(rows, "preference_role_label"),
        "actions": _counter_rows(rows, "action"),
    }


def _counter_rows(rows: Sequence[Mapping[str, Any]], key: str, *, limit: int = 100) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter(_counter_key(row.get(key)) for row in rows)
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def _counter_key(value: object) -> str:
    if value is None:
        return "<none>"
    text = str(value)
    return text if text else "<empty>"


def _conflicting_key_count(groups: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]]) -> int:
    return sum(1 for group in groups.values() if len({int(row["action"]) for row in group}) > 1)


def _ambiguous_key_rows(
    groups: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0], len(item[1]))):
        action_counts = _counter_rows(group, "action")
        if len(action_counts) <= 1:
            continue
        rows.append(
            {
                "key": list(key),
                "row_count": len(group),
                "action_counts": action_counts,
                "summary": _row_collection_summary(group),
                "examples": [_row_example(row) for row in group[:max_examples]],
            }
        )
        if len(rows) >= max_examples:
            break
    return rows


def _source_key(row: Mapping[str, Any]) -> tuple[str, int | None]:
    return (str(row.get("source_opponent_policy_id") or ""), _optional_int(row.get("source_pair_index")))


def _row_example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": _optional_int(row.get("step_index")),
        "episode_index": _optional_int(row.get("episode_index")),
        "action": _optional_int(row.get("action")),
        "source_opponent_policy_id": str(row.get("source_opponent_policy_id") or ""),
        "opponent_context_index": _optional_int(row.get("opponent_context_index")),
        "source_pair_index": _optional_int(row.get("source_pair_index")),
        "preference_pair_id": _optional_int(row.get("preference_pair_id")),
        "preference_role": _optional_int(row.get("preference_role")),
        "preference_role_label": str(row.get("preference_role_label") or ""),
        "source_dataset_label": str(row.get("source_dataset_label") or ""),
        "current_state_hash": str(row.get("current_state_hash") or ""),
        "history_hash": str(row.get("history_hash") or ""),
        "legal_candidate_hash": str(row.get("legal_candidate_hash") or ""),
    }


def _source_dataset_label(bundle: Mapping[str, Any]) -> str:
    source_label = str(bundle.get("source_dataset_label") or "").strip()
    if source_label:
        return source_label
    merge_label = str(bundle.get("merge_source_dataset_label") or "").strip()
    return merge_label


def _opponent_context_policy_ids(config: PairedOutcomePreferenceSurfacePrototypeConfig) -> tuple[str, ...]:
    policy_ids = [str(policy_id).strip() for policy_id in config.opponent_context_policy_ids if str(policy_id).strip()]
    if config.stack_config_path is not None:
        stack = load_stack_config(config.stack_config_path)
        policy_ids.extend(str(policy_id).strip() for policy_id in stack.config.model.opponent_context_policy_ids)
    result: list[str] = []
    seen: set[str] = set()
    for policy_id in policy_ids:
        if policy_id and policy_id not in seen:
            seen.add(policy_id)
            result.append(policy_id)
    return tuple(result)


def _context_index_for_policy_id(policy_id: str, context_index_by_policy_id: Mapping[str, int]) -> int:
    policy_text = str(policy_id).strip()
    if not policy_text:
        return 0
    exact = context_index_by_policy_id.get(policy_text)
    if exact is not None:
        return int(exact)
    for configured_policy_id, configured_index in context_index_by_policy_id.items():
        if policy_text.endswith(f"_{configured_policy_id}"):
            return int(configured_index)
    return 0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PairedOutcomePreferenceSurfacePrototypeConfig",
    "build_paired_outcome_preference_surface_prototype_report",
    "write_paired_outcome_preference_surface_prototype_report",
]
