"""Focus-label selection rules for trajectory BC replay batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc_dataset_schema import ReplayTrajectoryDataset


@dataclass(slots=True)
class TrajectoryBcReplayFocusGroupState:
    name: str
    source_labels: tuple[str, ...]
    fraction: float
    indices: np.ndarray
    order: np.ndarray
    cursor: int = 0
    last_episode_count: int = 0


def episode_indices_by_source_label(
    dataset: ReplayTrajectoryDataset,
    *,
    source_labels: tuple[str, ...],
    dataset_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not source_labels:
        return None, None
    labels = source_labels_by_episode(dataset)
    available = set(labels)
    missing = [label for label in source_labels if label not in available]
    if missing:
        raise ValueError(f"trajectory BC focus source labels not found in {dataset_path}: {', '.join(missing)}")
    focus_label_set = set(source_labels)
    focus = np.asarray(
        [index for index, label in enumerate(labels) if label in focus_label_set],
        dtype=np.int64,
    )
    nonfocus = np.asarray(
        [index for index, label in enumerate(labels) if label not in focus_label_set],
        dtype=np.int64,
    )
    if focus.size <= 0:
        raise ValueError(f"trajectory BC focus source labels selected no episodes in {dataset_path}")
    return focus, nonfocus


def focus_group_configs_from_structured_aux(structured_aux: Any) -> tuple[Any, ...]:
    raw_groups = getattr(structured_aux, "trajectory_bc_focus_groups", ())
    if raw_groups is None:
        return ()
    return tuple(raw_groups)


def focus_groups_by_source_label(
    dataset: ReplayTrajectoryDataset,
    *,
    focus_groups: tuple[Any, ...],
    dataset_path: Path,
    rng: np.random.Generator,
) -> tuple[tuple[TrajectoryBcReplayFocusGroupState, ...], np.ndarray | None]:
    if not focus_groups:
        return (), None
    labels = source_labels_by_episode(dataset)
    available = set(labels)
    claimed: set[str] = set()
    seen_names: set[str] = set()
    group_states: list[TrajectoryBcReplayFocusGroupState] = []
    total_fraction = 0.0
    for index, raw_group in enumerate(focus_groups):
        group_name = str(getattr(raw_group, "name", f"group_{index}")).strip() or f"group_{index}"
        if group_name in seen_names:
            raise ValueError(f"trajectory BC focus groups contain duplicate name: {group_name}")
        seen_names.add(group_name)
        source_labels = tuple(
            str(label).strip() for label in getattr(raw_group, "source_labels", ()) if str(label).strip()
        )
        if not source_labels:
            raise ValueError(f"trajectory BC focus group {group_name!r} must contain at least one source label")
        fraction = float(getattr(raw_group, "fraction", 0.0))
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"trajectory BC focus group {group_name!r} fraction must be between 0.0 and 1.0")
        total_fraction += fraction
        if total_fraction > 1.0 + 1e-9:
            raise ValueError("trajectory BC focus group fractions must sum to <= 1.0")
        missing = [label for label in source_labels if label not in available]
        if missing:
            raise ValueError(
                f"trajectory BC focus group source labels not found in {dataset_path}: {', '.join(missing)}"
            )
        duplicate = sorted(label for label in source_labels if label in claimed)
        if duplicate:
            raise ValueError(
                "trajectory BC focus group source labels overlap across groups in "
                f"{dataset_path}: {', '.join(duplicate)}"
            )
        claimed.update(source_labels)
        indices = np.asarray(
            [episode_index for episode_index, label in enumerate(labels) if label in set(source_labels)],
            dtype=np.int64,
        )
        if indices.size <= 0:
            raise ValueError(f"trajectory BC focus group {group_name!r} selected no episodes in {dataset_path}")
        order = rng.permutation(indices)
        group_states.append(
            TrajectoryBcReplayFocusGroupState(
                name=group_name,
                source_labels=source_labels,
                fraction=fraction,
                indices=indices,
                order=order,
            )
        )
    focus_labels = set(claimed)
    nonfocus = np.asarray(
        [index for index, label in enumerate(labels) if label not in focus_labels],
        dtype=np.int64,
    )
    return tuple(group_states), nonfocus


def focus_group_counts(*, batch_size: int, target_focus_count: int, fractions: tuple[float, ...]) -> tuple[int, ...]:
    if not fractions or target_focus_count <= 0:
        return tuple(0 for _ in fractions)
    raw_counts = [float(batch_size) * float(fraction) for fraction in fractions]
    counts = [int(np.floor(raw_count)) for raw_count in raw_counts]
    for index, fraction in enumerate(fractions):
        if fraction > 0.0 and counts[index] <= 0 and sum(counts) < target_focus_count:
            counts[index] = 1
    while sum(counts) < target_focus_count:
        remainders = [raw_count - int(np.floor(raw_count)) for raw_count in raw_counts]
        ranked_indices = sorted(
            range(len(counts)),
            key=lambda item: (remainders[item], fractions[item], -item),
            reverse=True,
        )
        for best_index in ranked_indices:
            if sum(counts) >= target_focus_count:
                break
            counts[best_index] += 1
    while sum(counts) > target_focus_count:
        best_index = max(range(len(counts)), key=lambda item: (counts[item], -fractions[item], -item))
        counts[best_index] -= 1
    return tuple(int(count) for count in counts)


def source_labels_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return ["" for _ in range(int(dataset.episode_count))]
    labels: list[str] = []
    for bundle in bundles:
        label = bundle.get("source_dataset_label") if isinstance(bundle, dict) else None
        labels.append(str(label or ""))
    return labels


def take_from_order(
    *,
    order: np.ndarray,
    cursor: int,
    count: int,
    source_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray, int]:
    taken: list[int] = []
    active_order = order
    active_cursor = int(cursor)
    while len(taken) < int(count):
        if active_cursor >= int(active_order.shape[0]):
            active_order = rng.permutation(source_indices)
            active_cursor = 0
        remaining = int(count) - len(taken)
        end = min(active_cursor + remaining, int(active_order.shape[0]))
        taken.extend(int(index) for index in active_order[active_cursor:end].astype(np.int64).tolist())
        active_cursor = end
    return taken, active_order, active_cursor


def metric_key_fragment(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "group"


__all__ = [
    "TrajectoryBcReplayFocusGroupState",
    "episode_indices_by_source_label",
    "focus_group_configs_from_structured_aux",
    "focus_group_counts",
    "focus_groups_by_source_label",
    "metric_key_fragment",
    "source_labels_by_episode",
    "take_from_order",
]
