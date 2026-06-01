"""Dataset construction for paired-outcome contrastive replay."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.experiments.paired_outcome_contrastive_sources import PairedOutcomeContrastiveSource
from weiss_rl.experiments.teacher_action_overrides import (
    TeacherActionOverrideExportConfig,
    build_teacher_action_overrides_from_inspections,
)
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    merge_replay_trajectory_bc_datasets,
    replay_trajectory_bc_batch,
    save_replay_trajectory_bc_dataset,
)
from weiss_rl.training.paired_swing_replay import paired_swing_distinct_train_row_count

SCRIPT_KIND = "paired_outcome_contrastive_dataset_v1"


@dataclass(frozen=True, slots=True)
class PairedOutcomeContrastiveBuildConfig:
    sources: tuple[PairedOutcomeContrastiveSource, ...]
    output_dataset: Path
    min_total_variation: float = 0.0
    max_rows_per_bundle: int | None = None
    max_rows: int | None = None
    positive_action_source: str = "actions"
    negative_action_source: str = "teacher_action"


def build_paired_outcome_contrastive_dataset(
    config: PairedOutcomeContrastiveBuildConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Build a paired-swing dataset where recorded winner actions beat losing-policy top actions."""

    if not config.sources:
        raise ValueError("sources must contain at least one source")
    if str(config.positive_action_source).strip().lower() != "actions":
        raise ValueError("paired-outcome contrastive datasets require positive_action_source='actions'")
    if str(config.negative_action_source).strip().lower() != "teacher_action":
        raise ValueError("paired-outcome contrastive datasets require negative_action_source='teacher_action'")

    datasets: list[ReplayTrajectoryDataset] = []
    labels: list[str] = []
    source_summaries: list[dict[str, Any]] = []
    for source in config.sources:
        dataset, source_summary = build_paired_outcome_contrastive_source_dataset(
            source,
            min_total_variation=float(config.min_total_variation),
            max_rows_per_bundle=config.max_rows_per_bundle,
            max_rows=config.max_rows,
        )
        datasets.append(dataset)
        labels.append(source.source_label)
        source_summaries.append(source_summary)

    merged = datasets[0] if len(datasets) == 1 else merge_replay_trajectory_bc_datasets(datasets, source_labels=labels)
    distinct_rows = paired_swing_distinct_train_row_count(
        merged,
        positive_action_source="actions",
        negative_action_source="teacher_action",
    )
    if distinct_rows <= 0:
        raise ValueError("paired-outcome contrastive dataset has no distinct train rows")
    merged.metadata["paired_outcome_contrastive_generation"] = {
        "kind": SCRIPT_KIND,
        "positive_action_source": "actions",
        "negative_action_source": "teacher_action",
        "min_total_variation": float(config.min_total_variation),
        "max_rows_per_bundle": config.max_rows_per_bundle,
        "max_rows": config.max_rows,
        "sources": source_summaries,
        "distinct_train_rows": int(distinct_rows),
    }
    merged.metadata["intended_auxiliary"] = "paired_swing_replay"
    save_replay_trajectory_bc_dataset(config.output_dataset, merged)
    return merged, {
        "kind": SCRIPT_KIND,
        "dataset_path": config.output_dataset.as_posix(),
        "dataset_metadata": dict(merged.metadata),
    }


def build_paired_outcome_contrastive_source_dataset(
    source: PairedOutcomeContrastiveSource,
    *,
    min_total_variation: float = 0.0,
    max_rows_per_bundle: int | None = None,
    max_rows: int | None = None,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Apply losing-policy top-action overrides to a single winner-trajectory dataset."""

    if not source.inspection_jsons:
        raise ValueError(f"source {source.source_label!r} has no inspection reports")
    dataset = load_replay_trajectory_bc_dataset(source.source_dataset_path)
    rows, override_summary = build_teacher_action_overrides_from_inspections(
        TeacherActionOverrideExportConfig(
            inspection_jsons=source.inspection_jsons,
            min_total_variation=float(min_total_variation),
            max_rows_per_bundle=max_rows_per_bundle,
            max_rows=max_rows,
        )
    )
    contrastive_dataset, apply_summary = apply_policy_b_top_action_overrides(
        dataset,
        override_rows=rows,
        source_label=source.source_label,
        source_role=source.source_role,
        source_dataset_path=source.source_dataset_path,
        source_opponent_policy_id=source.source_opponent_policy_id,
    )
    summary = {
        "source_label": source.source_label,
        "source_role": source.source_role,
        "source_opponent_policy_id": source.source_opponent_policy_id,
        "source_dataset_path": source.source_dataset_path.as_posix(),
        "inspection_jsons": [path.as_posix() for path in source.inspection_jsons],
        "override_summary": override_summary,
        "apply_summary": apply_summary,
        "train_rows": int(contrastive_dataset.metadata.get("train_rows", 0)),
        "bundle_count": int(contrastive_dataset.metadata.get("bundle_count", 0)),
    }
    return contrastive_dataset, summary


def apply_policy_b_top_action_overrides(
    dataset: ReplayTrajectoryDataset,
    *,
    override_rows: Sequence[Mapping[str, Any]],
    source_label: str,
    source_role: str,
    source_dataset_path: Path,
    source_opponent_policy_id: str = "",
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Return a dataset whose train rows are action-vs-policy-B-top contrasts."""

    if not override_rows:
        raise ValueError(f"no override rows provided for source {source_label!r}")

    work = _copy_dataset_arrays(dataset)
    work.policy_train_mask.fill(False)
    work.teacher_valid.fill(False)
    work.teacher_action.fill(-1)
    bundle_lookup = _episode_indices_by_bundle(dataset)
    counters: Counter[str] = Counter()
    train_rows_by_episode: Counter[int] = Counter()

    for row in override_rows:
        counters["override_rows"] += 1
        step_index = int(row["step_index"])
        teacher_action = int(row["teacher_action"])
        episode_index = _episode_index_for_override(row, bundle_lookup)
        if episode_index is None:
            counters["skipped_missing_bundle"] += 1
            continue
        if step_index < 0 or step_index >= int(work.time_steps):
            counters["skipped_step_out_of_range"] += 1
            continue
        if not bool(dataset.policy_train_mask[step_index, episode_index]):
            counters["skipped_nontrainable_source_row"] += 1
            continue
        recorded_action = int(dataset.actions[step_index, episode_index])
        if teacher_action == recorded_action:
            counters["skipped_same_action"] += 1
            continue
        if not _row_contains_legal_action(
            dataset, step_index=step_index, episode_index=episode_index, action=teacher_action
        ):
            counters["skipped_teacher_action_not_legal"] += 1
            continue

        work.teacher_action[step_index, episode_index] = teacher_action
        work.teacher_valid[step_index, episode_index] = True
        work.policy_train_mask[step_index, episode_index] = True
        train_rows_by_episode[episode_index] += 1
        counters["written_train_rows"] += 1

    keep_indices = tuple(sorted(train_rows_by_episode))
    if not keep_indices:
        raise ValueError(f"source {source_label!r} produced no train rows after contrastive filtering")

    selected_bundles = _annotated_selected_bundles(
        dataset,
        source_label=source_label,
        source_role=source_role,
        source_dataset_path=source_dataset_path,
        source_opponent_policy_id=source_opponent_policy_id,
        train_rows_by_episode=train_rows_by_episode,
        keep_indices=keep_indices,
    )
    subset = _subset_dataset(work, episode_indices=keep_indices, selected_bundles=selected_bundles)
    distinct_rows = paired_swing_distinct_train_row_count(
        subset,
        positive_action_source="actions",
        negative_action_source="teacher_action",
    )
    subset.metadata["paired_outcome_contrastive_source"] = {
        "kind": SCRIPT_KIND,
        "source_label": source_label,
        "source_role": source_role,
        "source_dataset_path": Path(source_dataset_path).as_posix(),
        "source_opponent_policy_id": source_opponent_policy_id,
        "positive_action_source": "actions",
        "negative_action_source": "teacher_action",
        "distinct_train_rows": int(distinct_rows),
        "counters": dict(sorted(counters.items())),
    }
    subset.metadata["intended_auxiliary"] = "paired_swing_replay"
    summary = {
        "source_label": source_label,
        "source_role": source_role,
        "source_dataset_path": Path(source_dataset_path).as_posix(),
        "source_opponent_policy_id": source_opponent_policy_id,
        "kept_episode_count": len(keep_indices),
        "train_rows": int(subset.metadata["train_rows"]),
        "distinct_train_rows": int(distinct_rows),
        "counters": dict(sorted(counters.items())),
    }
    return subset, summary


def write_paired_outcome_contrastive_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_dataset_arrays(dataset: ReplayTrajectoryDataset) -> ReplayTrajectoryDataset:
    return ReplayTrajectoryDataset(
        obs=np.array(dataset.obs, copy=True),
        actor=np.array(dataset.actor, copy=True),
        to_play_seat=np.array(dataset.to_play_seat, copy=True),
        actions=np.array(dataset.actions, copy=True),
        legal_ids=np.array(dataset.legal_ids, copy=True),
        legal_offsets=np.array(dataset.legal_offsets, copy=True),
        legal_action_meta=np.array(dataset.legal_action_meta, copy=True),
        teacher_family=np.array(dataset.teacher_family, copy=True),
        teacher_slot=np.array(dataset.teacher_slot, copy=True),
        teacher_move_source=np.array(dataset.teacher_move_source, copy=True),
        teacher_attack_type=np.array(dataset.teacher_attack_type, copy=True),
        teacher_action=np.array(dataset.teacher_action, copy=True),
        teacher_valid=np.array(dataset.teacher_valid, copy=True),
        policy_train_mask=np.array(dataset.policy_train_mask, copy=True),
        reset_before_step=np.array(dataset.reset_before_step, copy=True),
        metadata=dict(dataset.metadata),
    )


def _subset_dataset(
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: Sequence[int],
    selected_bundles: Sequence[Mapping[str, Any]],
) -> ReplayTrajectoryDataset:
    batch = replay_trajectory_bc_batch(dataset, episode_indices=episode_indices)
    metadata = dict(dataset.metadata)
    metadata["bundle_count"] = len(episode_indices)
    metadata["episode_count"] = len(episode_indices)
    metadata["requested_bundle_count"] = len(episode_indices)
    metadata["row_count"] = int(batch["obs"].shape[0] * batch["obs"].shape[1])
    metadata["time_steps"] = int(batch["obs"].shape[0])
    metadata["train_rows"] = int(np.count_nonzero(batch["policy_train_mask"]))
    metadata["teacher_valid_rows"] = int(np.count_nonzero(batch["teacher_valid"]))
    metadata["teacher_action_override_rows"] = int(np.count_nonzero(batch["policy_train_mask"]))
    metadata["selected_bundles"] = [dict(item) for item in selected_bundles]
    return ReplayTrajectoryDataset(
        obs=np.asarray(batch["obs"], dtype=np.float32),
        actor=np.asarray(batch["actor"]),
        to_play_seat=np.asarray(batch["to_play_seat"]),
        actions=np.asarray(batch["actions"]),
        legal_ids=np.asarray(batch["legal_ids"], dtype=np.uint32),
        legal_offsets=np.asarray(batch["legal_offsets"], dtype=np.uint32),
        legal_action_meta=np.asarray(batch["legal_action_meta"], dtype=np.uint16),
        teacher_family=np.asarray(batch["teacher_family"], dtype=np.int32),
        teacher_slot=np.asarray(batch["teacher_slot"], dtype=np.int32),
        teacher_move_source=np.asarray(batch["teacher_move_source"], dtype=np.int32),
        teacher_attack_type=np.asarray(batch["teacher_attack_type"], dtype=np.int32),
        teacher_action=np.asarray(batch["teacher_action"], dtype=np.int32),
        teacher_valid=np.asarray(batch["teacher_valid"], dtype=np.bool_),
        policy_train_mask=np.asarray(batch["policy_train_mask"], dtype=np.bool_),
        reset_before_step=np.asarray(batch["reset_before_step"], dtype=np.bool_),
        metadata=metadata,
    )


def _episode_indices_by_bundle(dataset: ReplayTrajectoryDataset) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    raw_bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(raw_bundles, list):
        raise ValueError("dataset metadata must contain selected_bundles")
    if len(raw_bundles) != int(dataset.episode_count):
        raise ValueError("selected_bundles length must match dataset episode_count")
    for episode_index, bundle in enumerate(raw_bundles):
        if not isinstance(bundle, Mapping):
            continue
        raw_path = bundle.get("bundle_path")
        if isinstance(raw_path, str) and raw_path.strip():
            path = Path(raw_path)
            lookup[(path.resolve().as_posix(), -1)] = int(episode_index)
            lookup[(path.name, -1)] = int(episode_index)
        raw_name = bundle.get("bundle_name")
        if isinstance(raw_name, str) and raw_name.strip():
            lookup[(raw_name, -1)] = int(episode_index)
    return lookup


def _episode_index_for_override(row: Mapping[str, Any], lookup: Mapping[tuple[str, int], int]) -> int | None:
    raw_path = row.get("bundle_path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
        for key in ((path.resolve().as_posix(), -1), (path.name, -1)):
            if key in lookup:
                return int(lookup[key])
    raw_name = row.get("bundle_name")
    if isinstance(raw_name, str) and raw_name.strip():
        key = (raw_name, -1)
        if key in lookup:
            return int(lookup[key])
    return None


def _row_contains_legal_action(
    dataset: ReplayTrajectoryDataset,
    *,
    step_index: int,
    episode_index: int,
    action: int,
) -> bool:
    row_index = int(step_index) * int(dataset.episode_count) + int(episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    row_ids = np.asarray(dataset.legal_ids[start:stop], dtype=np.int64)
    return bool(np.any(row_ids == int(action)))


def _annotated_selected_bundles(
    dataset: ReplayTrajectoryDataset,
    *,
    source_label: str,
    source_role: str,
    source_dataset_path: Path,
    source_opponent_policy_id: str,
    train_rows_by_episode: Mapping[int, int],
    keep_indices: Sequence[int],
) -> list[dict[str, Any]]:
    raw_bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(raw_bundles, list):
        raise ValueError("dataset metadata must contain selected_bundles")
    selected: list[dict[str, Any]] = []
    for episode_index in keep_indices:
        raw = raw_bundles[int(episode_index)]
        bundle = dict(raw) if isinstance(raw, Mapping) else {}
        bundle["source_dataset_label"] = source_label
        bundle["outcome_contrastive_role"] = source_role
        bundle["source_dataset_path"] = Path(source_dataset_path).as_posix()
        bundle["source_opponent_policy_id"] = source_opponent_policy_id
        bundle["contrastive_train_rows"] = int(train_rows_by_episode[int(episode_index)])
        selected.append(bundle)
    return selected


__all__ = [
    "PairedOutcomeContrastiveBuildConfig",
    "SCRIPT_KIND",
    "apply_policy_b_top_action_overrides",
    "build_paired_outcome_contrastive_dataset",
    "build_paired_outcome_contrastive_source_dataset",
    "write_paired_outcome_contrastive_summary",
]
