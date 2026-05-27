"""Filtering helpers for paired-swing replay datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    save_replay_trajectory_bc_dataset,
    subset_replay_trajectory_bc_dataset,
)
from weiss_rl.training.paired_swing_replay import (
    _normalize_action_source,
    paired_swing_distinct_train_row_count,
)


@dataclass(frozen=True, slots=True)
class PairedSwingEpisodeFilterConfig:
    dataset_path: Path
    output_dataset_path: Path
    source_pair_indices: tuple[int, ...] = ()
    source_labels: tuple[str, ...] = ()
    positive_action_source: str = "actions"
    negative_action_source: str = "teacher_action"
    require_distinct_train_rows: bool = True


def filter_paired_swing_dataset(
    config: PairedSwingEpisodeFilterConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Filter paired-swing replay episodes by source metadata and save the result."""

    dataset = load_replay_trajectory_bc_dataset(Path(config.dataset_path))
    selected_bundles = _selected_bundles(dataset)
    source_pair_indices = tuple(int(index) for index in config.source_pair_indices)
    source_labels = tuple(str(label).strip() for label in config.source_labels if str(label).strip())
    if not source_pair_indices and not source_labels:
        raise ValueError("at least one source_pair_index or source_label filter is required")

    kept_indices: list[int] = []
    counters: Counter[str] = Counter()
    for episode_index, bundle in enumerate(selected_bundles):
        counters["candidate_episodes"] += 1
        pair_match = not source_pair_indices or any(
            pair_index in source_pair_indices for pair_index in _bundle_source_pair_indices(bundle)
        )
        label_match = not source_labels or _bundle_source_label(bundle) in source_labels
        if pair_match and label_match:
            kept_indices.append(int(episode_index))
            counters["kept_episodes"] += 1
        elif not pair_match:
            counters["skipped_pair_index"] += 1
        elif not label_match:
            counters["skipped_source_label"] += 1

    if not kept_indices:
        raise ValueError(f"paired-swing filter selected no episodes from {config.dataset_path}")

    positive_source = _normalize_action_source(
        config.positive_action_source,
        field_name="positive_action_source",
    )
    negative_source = _normalize_action_source(
        config.negative_action_source,
        field_name="negative_action_source",
    )
    subset = subset_replay_trajectory_bc_dataset(
        dataset,
        episode_indices=kept_indices,
        metadata_updates={
            "paired_swing_filter": {
                "kind": "paired_swing_episode_filter_v1",
                "source_dataset_path": Path(config.dataset_path).as_posix(),
                "source_pair_indices": list(source_pair_indices),
                "source_labels": list(source_labels),
                "kept_episode_indices": kept_indices,
                "positive_action_source": positive_source,
                "negative_action_source": negative_source,
                "counters": dict(sorted(counters.items())),
            },
            "intended_auxiliary": "paired_swing_replay",
        },
    )
    distinct_rows = paired_swing_distinct_train_row_count(
        subset,
        positive_action_source=positive_source,
        negative_action_source=negative_source,
    )
    if bool(config.require_distinct_train_rows) and distinct_rows <= 0:
        raise ValueError(f"paired-swing filter produced no distinct train rows: {config.output_dataset_path}")
    subset.metadata["paired_swing_filter"]["distinct_train_rows"] = int(distinct_rows)
    save_replay_trajectory_bc_dataset(Path(config.output_dataset_path), subset)
    summary = {
        "kind": "paired_swing_episode_filter_v1",
        "source_dataset_path": Path(config.dataset_path).as_posix(),
        "output_dataset_path": Path(config.output_dataset_path).as_posix(),
        "source_pair_indices": list(source_pair_indices),
        "source_labels": list(source_labels),
        "kept_episode_indices": kept_indices,
        "episode_count": int(subset.episode_count),
        "train_rows": int(subset.metadata.get("train_rows", 0)),
        "distinct_train_rows": int(distinct_rows),
        "positive_action_source": positive_source,
        "negative_action_source": negative_source,
        "counters": dict(sorted(counters.items())),
    }
    return subset, summary


def write_paired_swing_filter_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_bundles(dataset: ReplayTrajectoryDataset) -> list[Mapping[str, Any]]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list):
        raise ValueError("dataset metadata must contain selected_bundles")
    if len(bundles) != int(dataset.episode_count):
        raise ValueError("selected_bundles length must match dataset episode_count")
    return [bundle if isinstance(bundle, Mapping) else {} for bundle in bundles]


def _bundle_source_pair_indices(bundle: Mapping[str, Any]) -> tuple[int, ...]:
    raw_indices = bundle.get("source_pair_indices")
    if isinstance(raw_indices, (list, tuple)):
        return tuple(int(index) for index in raw_indices)
    raw_index = bundle.get("source_pair_index")
    if raw_index is not None and str(raw_index).strip():
        return (int(raw_index),)
    return ()


def _bundle_source_label(bundle: Mapping[str, Any]) -> str:
    return str(bundle.get("source_dataset_label") or bundle.get("source_label") or "").strip()


__all__ = [
    "PairedSwingEpisodeFilterConfig",
    "filter_paired_swing_dataset",
    "write_paired_swing_filter_summary",
]
