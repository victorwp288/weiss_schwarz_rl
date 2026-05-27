"""Episode-level filters for paired-outcome preference replay datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    save_replay_trajectory_bc_dataset,
    subset_replay_trajectory_bc_dataset,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceFilterConfig:
    dataset_path: Path
    output_dataset_path: Path
    output_summary_json: Path | None = None
    include_preference_pair_ids: tuple[int, ...] = ()
    exclude_preference_pair_ids: tuple[int, ...] = ()
    include_source_pair_indices: tuple[int, ...] = ()
    exclude_source_pair_indices: tuple[int, ...] = ()
    include_source_opponent_policy_ids: tuple[str, ...] = ()
    exclude_source_opponent_policy_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceRowFilterConfig:
    dataset_path: Path
    output_dataset_path: Path
    output_summary_json: Path | None = None
    require_action_difference: bool = True
    exclude_same_family_action_differences: bool = False
    require_same_current_state: bool = False
    require_same_history: bool = False
    exclude_current_state_conflicts: bool = False
    exclude_history_conflicts: bool = False


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSpanFilterConfig:
    dataset_path: Path
    span_audit_json: Path
    output_dataset_path: Path
    output_summary_json: Path | None = None
    include_span_modes: tuple[str, ...] = ("repeated_action_label", "repeated_family")
    require_audit_pass: bool = True
    keep_span_fill_rows: bool = False


def filter_paired_outcome_preference_dataset(
    config: PairedOutcomePreferenceFilterConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    bundles = _selected_bundles(dataset)
    selected_indices = select_preference_episode_indices(
        bundles=bundles,
        include_preference_pair_ids=config.include_preference_pair_ids,
        exclude_preference_pair_ids=config.exclude_preference_pair_ids,
        include_source_pair_indices=config.include_source_pair_indices,
        exclude_source_pair_indices=config.exclude_source_pair_indices,
        include_source_opponent_policy_ids=config.include_source_opponent_policy_ids,
        exclude_source_opponent_policy_ids=config.exclude_source_opponent_policy_ids,
    )
    if not selected_indices:
        raise ValueError(f"preference filter selected no episodes from {config.dataset_path}")
    selected_bundles = [dict(bundles[index]) for index in selected_indices]
    subset = subset_replay_trajectory_bc_dataset(
        dataset,
        episode_indices=selected_indices,
        selected_bundles=selected_bundles,
        metadata_updates={
            "paired_outcome_preference_filter": {
                "kind": "paired_outcome_preference_episode_filter_v1",
                "source_dataset_path": config.dataset_path.as_posix(),
                "include_preference_pair_ids": list(config.include_preference_pair_ids),
                "exclude_preference_pair_ids": list(config.exclude_preference_pair_ids),
                "include_source_pair_indices": list(config.include_source_pair_indices),
                "exclude_source_pair_indices": list(config.exclude_source_pair_indices),
                "include_source_opponent_policy_ids": list(config.include_source_opponent_policy_ids),
                "exclude_source_opponent_policy_ids": list(config.exclude_source_opponent_policy_ids),
                "selected_episode_indices": selected_indices,
            }
        },
    )
    if int(subset.metadata.get("train_rows", 0)) <= 0:
        raise ValueError(f"preference filter produced no train rows: {config.output_dataset_path}")
    save_replay_trajectory_bc_dataset(config.output_dataset_path, subset)
    summary = {
        "kind": "paired_outcome_preference_episode_filter_v1",
        "source_dataset_path": config.dataset_path.as_posix(),
        "output_dataset_path": config.output_dataset_path.as_posix(),
        "input_episode_count": int(dataset.episode_count),
        "output_episode_count": int(subset.episode_count),
        "input_train_rows": int(dataset.metadata.get("train_rows", 0)),
        "output_train_rows": int(subset.metadata.get("train_rows", 0)),
        "selected_episode_indices": selected_indices,
        "selected_preference_pair_ids": _unique_int_bundle_values(selected_bundles, "preference_pair_id"),
        "selected_source_pair_indices": _unique_source_pair_indices(selected_bundles),
        "selected_source_opponent_policy_ids": _unique_source_opponent_policy_ids(selected_bundles),
        "include_preference_pair_ids": list(config.include_preference_pair_ids),
        "exclude_preference_pair_ids": list(config.exclude_preference_pair_ids),
        "include_source_pair_indices": list(config.include_source_pair_indices),
        "exclude_source_pair_indices": list(config.exclude_source_pair_indices),
        "include_source_opponent_policy_ids": list(config.include_source_opponent_policy_ids),
        "exclude_source_opponent_policy_ids": list(config.exclude_source_opponent_policy_ids),
    }
    if config.output_summary_json is not None:
        config.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return subset, summary


def filter_paired_outcome_preference_rows(
    config: PairedOutcomePreferenceRowFilterConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    row_mask, pair_summaries = aligned_preference_pair_row_mask(
        dataset,
        require_action_difference=bool(config.require_action_difference),
        exclude_same_family_action_differences=bool(config.exclude_same_family_action_differences),
        require_same_current_state=bool(config.require_same_current_state),
        require_same_history=bool(config.require_same_history),
        exclude_current_state_conflicts=bool(config.exclude_current_state_conflicts),
        exclude_history_conflicts=bool(config.exclude_history_conflicts),
    )
    train_rows = int(np.count_nonzero(row_mask))
    if train_rows <= 0:
        raise ValueError(f"row filter produced no train rows: {config.output_dataset_path}")
    metadata = dict(dataset.metadata)
    metadata["train_rows"] = train_rows
    metadata["teacher_action_override_rows"] = train_rows
    metadata["paired_outcome_preference_row_filter"] = {
        "kind": "paired_outcome_preference_row_filter_v1",
        "source_dataset_path": config.dataset_path.as_posix(),
        "require_action_difference": bool(config.require_action_difference),
        "exclude_same_family_action_differences": bool(config.exclude_same_family_action_differences),
        "require_same_current_state": bool(config.require_same_current_state),
        "require_same_history": bool(config.require_same_history),
        "exclude_current_state_conflicts": bool(config.exclude_current_state_conflicts),
        "exclude_history_conflicts": bool(config.exclude_history_conflicts),
        "input_train_rows": int(dataset.metadata.get("train_rows", 0)),
        "output_train_rows": train_rows,
        "pair_summaries": pair_summaries,
    }
    filtered = ReplayTrajectoryDataset(
        obs=dataset.obs,
        actor=dataset.actor,
        to_play_seat=dataset.to_play_seat,
        actions=dataset.actions,
        legal_ids=dataset.legal_ids,
        legal_offsets=dataset.legal_offsets,
        legal_action_meta=dataset.legal_action_meta,
        teacher_family=dataset.teacher_family,
        teacher_slot=dataset.teacher_slot,
        teacher_move_source=dataset.teacher_move_source,
        teacher_attack_type=dataset.teacher_attack_type,
        teacher_action=dataset.teacher_action,
        teacher_valid=dataset.teacher_valid,
        policy_train_mask=row_mask,
        reset_before_step=dataset.reset_before_step,
        metadata=metadata,
    )
    save_replay_trajectory_bc_dataset(config.output_dataset_path, filtered)
    summary = {
        "kind": "paired_outcome_preference_row_filter_v1",
        "source_dataset_path": config.dataset_path.as_posix(),
        "output_dataset_path": config.output_dataset_path.as_posix(),
        "require_action_difference": bool(config.require_action_difference),
        "exclude_same_family_action_differences": bool(config.exclude_same_family_action_differences),
        "require_same_current_state": bool(config.require_same_current_state),
        "require_same_history": bool(config.require_same_history),
        "exclude_current_state_conflicts": bool(config.exclude_current_state_conflicts),
        "exclude_history_conflicts": bool(config.exclude_history_conflicts),
        "input_train_rows": int(dataset.metadata.get("train_rows", 0)),
        "output_train_rows": train_rows,
        "pair_summaries": pair_summaries,
    }
    if config.output_summary_json is not None:
        config.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return filtered, summary


def filter_paired_outcome_preference_spans(
    config: PairedOutcomePreferenceSpanFilterConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Mask replay loss onto compact spans selected by a span audit report."""

    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    span_audit = _read_json_object(config.span_audit_json)
    if config.require_audit_pass and not bool(span_audit.get("passed")):
        raise ValueError(f"span audit did not pass: {config.span_audit_json}")
    audit_dataset_path = span_audit.get("dataset_path")
    if (
        isinstance(audit_dataset_path, str)
        and audit_dataset_path
        and not _same_path(
            Path(audit_dataset_path),
            config.dataset_path,
        )
    ):
        raise ValueError(
            "span audit dataset_path does not match filter dataset: "
            f"{audit_dataset_path!r} vs {config.dataset_path.as_posix()!r}"
        )

    bundles = _selected_bundles(dataset)
    pair_roles = _preference_pair_roles(bundles)
    selected_spans = select_span_audit_spans(span_audit, include_span_modes=config.include_span_modes)
    if not selected_spans:
        raise ValueError(f"span filter selected no spans from {config.span_audit_json}")

    base_mask = np.asarray(dataset.policy_train_mask, dtype=np.bool_)
    selected_mask = np.zeros_like(base_mask, dtype=np.bool_)
    kept_span_rows: list[dict[str, Any]] = []
    skipped_span_rows: list[dict[str, Any]] = []
    pair_row_counts: Counter[int] = Counter()
    opponent_row_counts: Counter[str] = Counter()
    for span in selected_spans:
        pair_id = _optional_int(span.get("preference_pair_id"))
        if pair_id is None or pair_id not in pair_roles:
            skipped_span_rows.append({"reason": "missing_pair_roles", "span": _summary_span(span)})
            continue
        roles = pair_roles[pair_id]
        if 0 not in roles or 1 not in roles:
            skipped_span_rows.append({"reason": "incomplete_pair_roles", "span": _summary_span(span)})
            continue
        step_indices = _span_step_indices(span, keep_span_fill_rows=bool(config.keep_span_fill_rows))
        if not step_indices:
            skipped_span_rows.append({"reason": "empty_step_indices", "span": _summary_span(span)})
            continue
        before = int(np.count_nonzero(selected_mask))
        for episode_index in (int(roles[1]), int(roles[0])):
            for step_index in step_indices:
                if 0 <= int(step_index) < int(dataset.time_steps):
                    selected_mask[int(step_index), episode_index] |= bool(base_mask[int(step_index), episode_index])
        added = int(np.count_nonzero(selected_mask)) - before
        if added <= 0:
            skipped_span_rows.append({"reason": "span_added_no_train_rows", "span": _summary_span(span)})
            continue
        pair_row_counts[int(pair_id)] += added
        opponent_row_counts[str(span.get("source_opponent_policy_id") or "")] += added
        kept = _summary_span(span)
        kept["added_rows"] = added
        kept_span_rows.append(kept)

    train_rows = int(np.count_nonzero(selected_mask))
    if train_rows <= 0:
        raise ValueError(f"span filter produced no train rows: {config.output_dataset_path}")

    metadata = dict(dataset.metadata)
    metadata["train_rows"] = train_rows
    metadata["teacher_action_override_rows"] = train_rows
    metadata["paired_outcome_preference_span_filter"] = {
        "kind": "paired_outcome_preference_span_filter_v1",
        "source_dataset_path": config.dataset_path.as_posix(),
        "span_audit_json": config.span_audit_json.as_posix(),
        "include_span_modes": list(_normalized_span_modes(config.include_span_modes)),
        "require_audit_pass": bool(config.require_audit_pass),
        "keep_span_fill_rows": bool(config.keep_span_fill_rows),
        "input_train_rows": int(dataset.metadata.get("train_rows", 0)),
        "output_train_rows": train_rows,
        "selected_span_count": len(kept_span_rows),
        "skipped_span_count": len(skipped_span_rows),
        "kept_spans": kept_span_rows,
        "skipped_spans": skipped_span_rows,
    }
    filtered = ReplayTrajectoryDataset(
        obs=dataset.obs,
        actor=dataset.actor,
        to_play_seat=dataset.to_play_seat,
        actions=dataset.actions,
        legal_ids=dataset.legal_ids,
        legal_offsets=dataset.legal_offsets,
        legal_action_meta=dataset.legal_action_meta,
        teacher_family=dataset.teacher_family,
        teacher_slot=dataset.teacher_slot,
        teacher_move_source=dataset.teacher_move_source,
        teacher_attack_type=dataset.teacher_attack_type,
        teacher_action=dataset.teacher_action,
        teacher_valid=dataset.teacher_valid,
        policy_train_mask=selected_mask,
        reset_before_step=dataset.reset_before_step,
        metadata=metadata,
    )
    save_replay_trajectory_bc_dataset(config.output_dataset_path, filtered)
    summary = {
        "kind": "paired_outcome_preference_span_filter_v1",
        "source_dataset_path": config.dataset_path.as_posix(),
        "span_audit_json": config.span_audit_json.as_posix(),
        "output_dataset_path": config.output_dataset_path.as_posix(),
        "include_span_modes": list(_normalized_span_modes(config.include_span_modes)),
        "require_audit_pass": bool(config.require_audit_pass),
        "keep_span_fill_rows": bool(config.keep_span_fill_rows),
        "input_train_rows": int(dataset.metadata.get("train_rows", 0)),
        "output_train_rows": train_rows,
        "selected_span_count": len(kept_span_rows),
        "skipped_span_count": len(skipped_span_rows),
        "selected_preference_pair_ids": sorted(pair_row_counts),
        "selected_opponents": sorted(opponent_row_counts),
        "pair_row_counts": {str(key): int(value) for key, value in sorted(pair_row_counts.items())},
        "opponent_row_counts": {str(key): int(value) for key, value in sorted(opponent_row_counts.items())},
        "kept_spans": kept_span_rows,
        "skipped_spans": skipped_span_rows,
    }
    if config.output_summary_json is not None:
        config.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return filtered, summary


def aligned_preference_pair_row_mask(
    dataset: ReplayTrajectoryDataset,
    *,
    require_action_difference: bool = True,
    exclude_same_family_action_differences: bool = False,
    require_same_current_state: bool = False,
    require_same_history: bool = False,
    exclude_current_state_conflicts: bool = False,
    exclude_history_conflicts: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep rows from matched preferred/rejected episodes at aligned timesteps.

    Whole-trajectory preference replay can push many nondecisive same-family rows.
    This mask keeps the paired outcome objective on the timesteps where both roles
    have supervision, optionally requiring the two roles to choose different actions.
    """

    bundles = _selected_bundles(dataset)
    pair_roles = _preference_pair_roles(bundles)
    base_mask = np.asarray(dataset.policy_train_mask, dtype=np.bool_)
    actions = np.asarray(dataset.actions, dtype=np.int64)
    if base_mask.shape != actions.shape:
        raise ValueError("policy_train_mask and actions must have the same shape")
    teacher_family = np.asarray(dataset.teacher_family, dtype=np.int64)
    if teacher_family.shape != actions.shape:
        raise ValueError("teacher_family and actions must have the same shape")
    selected_mask = np.zeros_like(base_mask, dtype=np.bool_)
    summaries: list[dict[str, Any]] = []
    current_state_conflict_hashes: set[str] = set()
    history_conflict_hashes: set[str] = set()
    if exclude_current_state_conflicts:
        current_state_conflict_hashes = _preference_conflict_hashes(
            dataset,
            pair_roles=pair_roles,
            base_mask=base_mask,
            actions=actions,
            key_kind="current_state",
        )
    if exclude_history_conflicts:
        history_conflict_hashes = _preference_conflict_hashes(
            dataset,
            pair_roles=pair_roles,
            base_mask=base_mask,
            actions=actions,
            key_kind="history",
        )
    for pair_id in sorted(pair_roles):
        roles = pair_roles[pair_id]
        if 0 not in roles or 1 not in roles:
            continue
        preferred_episode = int(roles[1])
        rejected_episode = int(roles[0])
        aligned = base_mask[:, preferred_episode] & base_mask[:, rejected_episode]
        action_diff = actions[:, preferred_episode] != actions[:, rejected_episode]
        same_family_action_diff = (
            action_diff
            & (teacher_family[:, preferred_episode] >= 0)
            & (teacher_family[:, rejected_episode] >= 0)
            & (teacher_family[:, preferred_episode] == teacher_family[:, rejected_episode])
        )
        same_current_state = _same_current_state_steps(
            dataset,
            preferred_episode=preferred_episode,
            rejected_episode=rejected_episode,
        )
        same_history = _same_history_steps(
            dataset,
            preferred_episode=preferred_episode,
            rejected_episode=rejected_episode,
        )
        current_state_conflict_steps = _conflict_steps(
            dataset,
            preferred_episode=preferred_episode,
            rejected_episode=rejected_episode,
            conflict_hashes=current_state_conflict_hashes,
            key_kind="current_state",
        )
        history_conflict_steps = _conflict_steps(
            dataset,
            preferred_episode=preferred_episode,
            rejected_episode=rejected_episode,
            conflict_hashes=history_conflict_hashes,
            key_kind="history",
        )
        keep_steps = aligned.copy()
        if require_action_difference:
            keep_steps &= action_diff
        pre_same_family_keep_steps = keep_steps.copy()
        if exclude_same_family_action_differences:
            keep_steps &= ~same_family_action_diff
        if require_same_current_state:
            keep_steps &= same_current_state
        if require_same_history:
            keep_steps &= same_history
        pre_conflict_keep_steps = keep_steps.copy()
        if exclude_current_state_conflicts:
            keep_steps &= ~current_state_conflict_steps
        if exclude_history_conflicts:
            keep_steps &= ~history_conflict_steps
        selected_mask[:, preferred_episode] = keep_steps
        selected_mask[:, rejected_episode] = keep_steps
        kept_step_indices = np.nonzero(keep_steps)[0].astype(np.int64).tolist()
        preferred_bundle = bundles[preferred_episode]
        summaries.append(
            {
                "preference_pair_id": int(pair_id),
                "source_pair_index": _first_optional_int(preferred_bundle, ("source_pair_index", "pair_index")),
                "source_opponent_policy_id": str(preferred_bundle.get("source_opponent_policy_id") or ""),
                "preferred_episode_index": preferred_episode,
                "rejected_episode_index": rejected_episode,
                "preferred_input_train_rows": int(np.count_nonzero(base_mask[:, preferred_episode])),
                "rejected_input_train_rows": int(np.count_nonzero(base_mask[:, rejected_episode])),
                "aligned_train_steps": int(np.count_nonzero(aligned)),
                "aligned_action_difference_steps": int(np.count_nonzero(aligned & action_diff)),
                "aligned_action_difference_same_family_steps": int(np.count_nonzero(aligned & same_family_action_diff)),
                "aligned_action_difference_cross_family_steps": int(
                    np.count_nonzero(aligned & action_diff & ~same_family_action_diff)
                ),
                "aligned_same_current_state_steps": int(np.count_nonzero(aligned & same_current_state)),
                "aligned_same_history_steps": int(np.count_nonzero(aligned & same_history)),
                "aligned_action_difference_same_current_state_steps": int(
                    np.count_nonzero(aligned & action_diff & same_current_state)
                ),
                "aligned_action_difference_same_history_steps": int(
                    np.count_nonzero(aligned & action_diff & same_history)
                ),
                "excluded_current_state_conflict_steps": int(
                    np.count_nonzero(pre_conflict_keep_steps & current_state_conflict_steps)
                ),
                "excluded_history_conflict_steps": int(
                    np.count_nonzero(pre_conflict_keep_steps & history_conflict_steps)
                ),
                "excluded_same_family_action_difference_steps": int(
                    np.count_nonzero(pre_same_family_keep_steps & same_family_action_diff)
                    if exclude_same_family_action_differences
                    else 0
                ),
                "output_steps": int(len(kept_step_indices)),
                "output_rows": int(len(kept_step_indices) * 2),
                "first_output_step": None if not kept_step_indices else int(kept_step_indices[0]),
            }
        )
    return selected_mask, summaries


def select_preference_episode_indices(
    *,
    bundles: Sequence[Mapping[str, Any]],
    include_preference_pair_ids: Sequence[int] = (),
    exclude_preference_pair_ids: Sequence[int] = (),
    include_source_pair_indices: Sequence[int] = (),
    exclude_source_pair_indices: Sequence[int] = (),
    include_source_opponent_policy_ids: Sequence[str] = (),
    exclude_source_opponent_policy_ids: Sequence[str] = (),
) -> list[int]:
    include_pair_ids = {int(value) for value in include_preference_pair_ids}
    exclude_pair_ids = {int(value) for value in exclude_preference_pair_ids}
    include_source_pairs = {int(value) for value in include_source_pair_indices}
    exclude_source_pairs = {int(value) for value in exclude_source_pair_indices}
    include_opponents = _normalized_opponent_ids(include_source_opponent_policy_ids)
    exclude_opponents = _normalized_opponent_ids(exclude_source_opponent_policy_ids)
    selected: list[int] = []
    for index, bundle in enumerate(bundles):
        preference_pair_id = _optional_int(bundle.get("preference_pair_id"))
        source_pair_index = _first_optional_int(bundle, ("source_pair_index", "pair_index"))
        opponent_policy_id = _source_opponent_policy_id(bundle)
        if include_pair_ids and preference_pair_id not in include_pair_ids:
            continue
        if preference_pair_id in exclude_pair_ids:
            continue
        if include_source_pairs and source_pair_index not in include_source_pairs:
            continue
        if source_pair_index in exclude_source_pairs:
            continue
        if include_opponents and opponent_policy_id not in include_opponents:
            continue
        if opponent_policy_id in exclude_opponents:
            continue
        selected.append(int(index))
    return selected


def select_span_audit_spans(
    span_audit: Mapping[str, Any],
    *,
    include_span_modes: Sequence[str],
) -> list[dict[str, Any]]:
    modes = _normalized_span_modes(include_span_modes)
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}

    def add_span(span: object, reason: str) -> None:
        if not isinstance(span, Mapping):
            return
        pair_id = _optional_int(span.get("preference_pair_id"))
        start = _optional_int(span.get("start_step"))
        end = _optional_int(span.get("end_step"))
        if pair_id is None or start is None or end is None:
            return
        key = (int(pair_id), int(start), int(end))
        row = selected.setdefault(key, dict(span))
        reasons = row.setdefault("selection_reasons", [])
        if isinstance(reasons, list) and reason not in reasons:
            reasons.append(reason)

    pair_summaries = span_audit.get("pair_summaries")
    if isinstance(pair_summaries, list):
        for row in pair_summaries:
            if not isinstance(row, Mapping):
                continue
            if "earliest" in modes:
                add_span(row.get("earliest_span"), "earliest")
            if "densest" in modes:
                add_span(row.get("densest_span"), "densest")

    candidate_spans = span_audit.get("candidate_spans")
    if isinstance(candidate_spans, list):
        if "all_compact" in modes:
            for span in candidate_spans:
                if isinstance(span, Mapping) and bool(span.get("compact")):
                    add_span(span, "all_compact")
        repeated_specs = (
            ("repeated_action_label", "repeated_action_label_edges", "action_label_edge_counts"),
            ("repeated_family", "repeated_family_edges", "family_edge_counts"),
            ("repeated_raw_action", "repeated_raw_action_edges", "raw_action_edge_counts"),
        )
        for mode, report_key, span_count_key in repeated_specs:
            if mode not in modes:
                continue
            repeated_keys = _repeated_edge_keys(span_audit.get(report_key))
            if not repeated_keys:
                continue
            for span in candidate_spans:
                if not isinstance(span, Mapping) or not bool(span.get("compact")):
                    continue
                if _span_count_keys(span.get(span_count_key)) & repeated_keys:
                    add_span(span, mode)

    return sorted(
        selected.values(),
        key=lambda span: (
            _optional_int(span.get("preference_pair_id")) or -1,
            _optional_int(span.get("start_step")) or -1,
            _optional_int(span.get("end_step")) or -1,
        ),
    )


def _selected_bundles(dataset: ReplayTrajectoryDataset) -> list[Mapping[str, Any]]:
    raw = dataset.metadata.get("selected_bundles")
    if not isinstance(raw, list) or len(raw) != int(dataset.episode_count):
        raise ValueError("preference dataset is missing selected_bundles metadata")
    return [bundle if isinstance(bundle, Mapping) else {} for bundle in raw]


def _preference_pair_roles(bundles: Sequence[Mapping[str, Any]]) -> dict[int, dict[int, int]]:
    roles_by_pair: dict[int, dict[int, int]] = {}
    for episode_index, bundle in enumerate(bundles):
        preference_pair_id = _optional_int(bundle.get("preference_pair_id"))
        preference_role = _optional_int(bundle.get("preference_role"))
        if preference_pair_id is None or preference_role not in (0, 1):
            continue
        roles = roles_by_pair.setdefault(int(preference_pair_id), {})
        if int(preference_role) in roles:
            raise ValueError(
                "paired outcome preference row filter requires at most one episode per "
                f"pair/role, got pair {preference_pair_id} role {preference_role}"
            )
        roles[int(preference_role)] = int(episode_index)
    return roles_by_pair


def _normalized_span_modes(raw_modes: Sequence[str]) -> tuple[str, ...]:
    allowed = {
        "earliest",
        "densest",
        "all_compact",
        "repeated_action_label",
        "repeated_family",
        "repeated_raw_action",
    }
    modes: list[str] = []
    for item in raw_modes:
        mode = str(item).strip().lower().replace("-", "_")
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"unknown span selection mode {item!r}; expected one of {sorted(allowed)}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        modes.extend(["repeated_action_label", "repeated_family"])
    return tuple(modes)


def _repeated_edge_keys(rows: object) -> set[str]:
    if not isinstance(rows, list):
        return set()
    keys: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            key = str(row.get("key") or "").strip()
            if key:
                keys.add(key)
    return keys


def _span_count_keys(rows: object) -> set[str]:
    if not isinstance(rows, list):
        return set()
    keys: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            key = str(row.get("key") or "").strip()
            if key:
                keys.add(key)
    return keys


def _span_step_indices(span: Mapping[str, Any], *, keep_span_fill_rows: bool) -> list[int]:
    start = _optional_int(span.get("start_step"))
    end = _optional_int(span.get("end_step"))
    if start is None or end is None:
        return []
    if keep_span_fill_rows:
        return list(range(int(start), int(end) + 1))
    raw_steps = span.get("edge_step_indices")
    if isinstance(raw_steps, list):
        steps = sorted(int(step) for step in {_optional_int(value) for value in raw_steps} if step is not None)
        return [int(step) for step in steps if step is not None and int(start) <= int(step) <= int(end)]
    edge_steps: list[int] = []
    edges = span.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, Mapping):
                step = _optional_int(edge.get("step_index"))
                if step is not None and int(start) <= int(step) <= int(end):
                    edge_steps.append(int(step))
    if edge_steps:
        return sorted(set(edge_steps))
    return list(range(int(start), int(end) + 1))


def _summary_span(span: Mapping[str, Any]) -> dict[str, Any]:
    raw_reasons = span.get("selection_reasons")
    reasons = list(raw_reasons) if isinstance(raw_reasons, list) else []
    return {
        "preference_pair_id": _optional_int(span.get("preference_pair_id")),
        "source_opponent_policy_id": str(span.get("source_opponent_policy_id") or ""),
        "source_pair_index": _optional_int(span.get("source_pair_index")),
        "start_step": _optional_int(span.get("start_step")),
        "end_step": _optional_int(span.get("end_step")),
        "span_width": _optional_int(span.get("span_width")),
        "different_action_count": _optional_int(span.get("different_action_count")),
        "primary_action_label_edge": str(span.get("primary_action_label_edge") or ""),
        "primary_family_edge": str(span.get("primary_family_edge") or ""),
        "primary_raw_action_edge": str(span.get("primary_raw_action_edge") or ""),
        "selection_reasons": reasons,
        "edge_step_indices": _span_step_indices(span, keep_span_fill_rows=False),
    }


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.as_posix() == right.as_posix()


def _unique_int_bundle_values(bundles: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    values = {_optional_int(bundle.get(key)) for bundle in bundles}
    return sorted(int(value) for value in values if value is not None)


def _unique_source_pair_indices(bundles: Sequence[Mapping[str, Any]]) -> list[int]:
    values = {_first_optional_int(bundle, ("source_pair_index", "pair_index")) for bundle in bundles}
    return sorted(int(value) for value in values if value is not None)


def _unique_source_opponent_policy_ids(bundles: Sequence[Mapping[str, Any]]) -> list[str]:
    values = {_source_opponent_policy_id(bundle) for bundle in bundles}
    return sorted(value for value in values if value)


def _normalized_opponent_ids(values: Sequence[str]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _source_opponent_policy_id(bundle: Mapping[str, Any]) -> str:
    return str(bundle.get("source_opponent_policy_id") or bundle.get("opponent_policy_id") or "").strip()


def _same_current_state_steps(
    dataset: ReplayTrajectoryDataset,
    *,
    preferred_episode: int,
    rejected_episode: int,
) -> np.ndarray:
    result = np.zeros((int(dataset.time_steps),), dtype=np.bool_)
    for step in range(int(dataset.time_steps)):
        result[step] = _state_hash(dataset, step_index=step, episode_index=preferred_episode) == _state_hash(
            dataset,
            step_index=step,
            episode_index=rejected_episode,
        )
    return result


def _same_history_steps(
    dataset: ReplayTrajectoryDataset,
    *,
    preferred_episode: int,
    rejected_episode: int,
) -> np.ndarray:
    result = np.zeros((int(dataset.time_steps),), dtype=np.bool_)
    for step in range(int(dataset.time_steps)):
        result[step] = _history_hash(dataset, step_index=step, episode_index=preferred_episode) == _history_hash(
            dataset,
            step_index=step,
            episode_index=rejected_episode,
        )
    return result


def _preference_conflict_hashes(
    dataset: ReplayTrajectoryDataset,
    *,
    pair_roles: Mapping[int, Mapping[int, int]],
    base_mask: np.ndarray,
    actions: np.ndarray,
    key_kind: str,
) -> set[str]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for roles in pair_roles.values():
        if 0 not in roles or 1 not in roles:
            continue
        preferred_episode = int(roles[1])
        rejected_episode = int(roles[0])
        aligned_diff = (
            base_mask[:, preferred_episode]
            & base_mask[:, rejected_episode]
            & (actions[:, preferred_episode] != actions[:, rejected_episode])
        )
        for step in np.nonzero(aligned_diff)[0].astype(np.int64).tolist():
            preferred_hash = _row_hash(
                dataset, key_kind=key_kind, step_index=int(step), episode_index=preferred_episode
            )
            rejected_hash = _row_hash(dataset, key_kind=key_kind, step_index=int(step), episode_index=rejected_episode)
            if preferred_hash != rejected_hash:
                continue
            grouped.setdefault(preferred_hash, []).append(
                (int(actions[int(step), preferred_episode]), int(actions[int(step), rejected_episode]))
            )
    return {key for key, edges in grouped.items() if _action_edges_conflict(edges)}


def _action_edges_conflict(edges: Sequence[tuple[int, int]]) -> bool:
    preferred_actions = {int(preferred) for preferred, _rejected in edges}
    if len(preferred_actions) > 1:
        return True
    edge_set = {(int(preferred), int(rejected)) for preferred, rejected in edges}
    return any((rejected, preferred) in edge_set for preferred, rejected in edge_set if preferred != rejected)


def _conflict_steps(
    dataset: ReplayTrajectoryDataset,
    *,
    preferred_episode: int,
    rejected_episode: int,
    conflict_hashes: set[str],
    key_kind: str,
) -> np.ndarray:
    result = np.zeros((int(dataset.time_steps),), dtype=np.bool_)
    if not conflict_hashes:
        return result
    for step in range(int(dataset.time_steps)):
        preferred_hash = _row_hash(dataset, key_kind=key_kind, step_index=step, episode_index=preferred_episode)
        rejected_hash = _row_hash(dataset, key_kind=key_kind, step_index=step, episode_index=rejected_episode)
        result[step] = preferred_hash == rejected_hash and preferred_hash in conflict_hashes
    return result


def _row_hash(dataset: ReplayTrajectoryDataset, *, key_kind: str, step_index: int, episode_index: int) -> str:
    if key_kind == "current_state":
        return _state_hash(dataset, step_index=step_index, episode_index=episode_index)
    if key_kind == "history":
        return _history_hash(dataset, step_index=step_index, episode_index=episode_index)
    raise ValueError(f"unknown conflict key kind {key_kind!r}")


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


def _first_optional_int(bundle: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _optional_int(bundle.get(key))
        if value is not None:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PairedOutcomePreferenceFilterConfig",
    "PairedOutcomePreferenceRowFilterConfig",
    "PairedOutcomePreferenceSpanFilterConfig",
    "aligned_preference_pair_row_mask",
    "filter_paired_outcome_preference_dataset",
    "filter_paired_outcome_preference_rows",
    "filter_paired_outcome_preference_spans",
    "select_preference_episode_indices",
    "select_span_audit_spans",
]
