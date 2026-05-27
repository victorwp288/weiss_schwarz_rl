"""In-training replay trajectory behavior-cloning regularizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    replay_trajectory_bc_batch,
)


@dataclass(slots=True)
class TrajectoryBcReplayFocusGroupState:
    name: str
    source_labels: tuple[str, ...]
    fraction: float
    indices: np.ndarray
    order: np.ndarray
    cursor: int = 0
    last_episode_count: int = 0


@dataclass(slots=True)
class TrajectoryBcReplayState:
    dataset: ReplayTrajectoryDataset
    rng: np.random.Generator
    batch_episodes: int
    aux_updates: int
    every_updates: int
    order: np.ndarray
    focus_source_labels: tuple[str, ...] = ()
    focus_fraction: float = 0.0
    focus_indices: np.ndarray | None = None
    nonfocus_indices: np.ndarray | None = None
    focus_order: np.ndarray | None = None
    nonfocus_order: np.ndarray | None = None
    focus_groups: tuple[TrajectoryBcReplayFocusGroupState, ...] = ()
    cursor: int = 0
    focus_cursor: int = 0
    nonfocus_cursor: int = 0
    last_focus_episode_count: int = 0
    last_nonfocus_episode_count: int = 0

    @classmethod
    def from_training_config(cls, training_config: Any, *, repo_root: Path) -> TrajectoryBcReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "trajectory_bc_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "trajectory_bc_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        dataset_path = Path(dataset_path_text)
        if not dataset_path.is_absolute():
            dataset_path = Path(repo_root) / dataset_path
        dataset = load_replay_trajectory_bc_dataset(dataset_path)
        if int(dataset.metadata.get("train_rows", 0)) <= 0:
            raise ValueError(f"trajectory BC dataset has no trainable rows: {dataset_path}")
        rng = np.random.default_rng(int(getattr(structured_aux, "trajectory_bc_seed", 20260516)))
        order = rng.permutation(dataset.episode_count)
        focus_source_labels = tuple(
            str(label).strip()
            for label in getattr(structured_aux, "trajectory_bc_focus_source_labels", ())
            if str(label).strip()
        )
        focus_fraction = float(getattr(structured_aux, "trajectory_bc_focus_fraction", 0.0))
        if focus_fraction < 0.0 or focus_fraction > 1.0:
            raise ValueError("trajectory_bc_focus_fraction must be between 0.0 and 1.0")
        focus_group_configs = _focus_group_configs_from_structured_aux(structured_aux)
        focus_groups, grouped_nonfocus_indices = _focus_groups_by_source_label(
            dataset,
            focus_groups=focus_group_configs,
            dataset_path=dataset_path,
            rng=rng,
        )
        if focus_groups:
            focus_labels = tuple(label for group in focus_groups for label in group.source_labels)
            focus_source_labels = focus_labels
            focus_fraction = sum(group.fraction for group in focus_groups)
            focus_indices = np.concatenate([group.indices for group in focus_groups], axis=0)
            nonfocus_indices = grouped_nonfocus_indices
            focus_order = None
            nonfocus_order = None if nonfocus_indices is None else rng.permutation(nonfocus_indices)
        else:
            focus_indices, nonfocus_indices = _episode_indices_by_source_label(
                dataset,
                source_labels=focus_source_labels,
                dataset_path=dataset_path,
            )
            focus_order = None if focus_indices is None else rng.permutation(focus_indices)
            nonfocus_order = None if nonfocus_indices is None else rng.permutation(nonfocus_indices)
        return cls(
            dataset=dataset,
            rng=rng,
            batch_episodes=int(getattr(structured_aux, "trajectory_bc_batch_episodes", 8)),
            aux_updates=int(getattr(structured_aux, "trajectory_bc_aux_updates", 1)),
            every_updates=every_updates,
            order=order,
            focus_source_labels=focus_source_labels,
            focus_fraction=focus_fraction,
            focus_indices=focus_indices,
            nonfocus_indices=nonfocus_indices,
            focus_order=focus_order,
            nonfocus_order=nonfocus_order,
            focus_groups=focus_groups,
        )

    def next_episode_indices(self) -> list[int]:
        if self.focus_groups:
            return self._next_grouped_episode_indices()
        if self.focus_indices is not None and self.focus_fraction > 0.0:
            return self._next_stratified_episode_indices()
        if self.cursor >= int(self.order.shape[0]):
            self.order = self.rng.permutation(self.dataset.episode_count)
            self.cursor = 0
        end = min(self.cursor + int(self.batch_episodes), int(self.order.shape[0]))
        indices = self.order[self.cursor : end].astype(np.int64).tolist()
        self.cursor = end
        self.last_focus_episode_count = 0
        self.last_nonfocus_episode_count = len(indices)
        if indices:
            return indices
        return self.next_episode_indices()

    def _next_stratified_episode_indices(self) -> list[int]:
        batch_size = int(self.batch_episodes)
        if batch_size <= 0:
            raise ValueError("trajectory BC batch_episodes must be >= 1")
        focus_available = 0 if self.focus_indices is None else int(self.focus_indices.shape[0])
        nonfocus_available = 0 if self.nonfocus_indices is None else int(self.nonfocus_indices.shape[0])
        focus_count = min(focus_available, int(np.ceil(batch_size * float(self.focus_fraction))))
        if focus_count <= 0 and focus_available > 0:
            focus_count = 1
        nonfocus_count = batch_size - focus_count
        if nonfocus_available <= 0:
            focus_count = batch_size
            nonfocus_count = 0
        elif nonfocus_count > nonfocus_available and focus_available > focus_count:
            extra_focus = min(focus_available - focus_count, nonfocus_count - nonfocus_available)
            focus_count += extra_focus
            nonfocus_count -= extra_focus
        focus = self._take_focus_episodes(focus_count)
        nonfocus = self._take_nonfocus_episodes(nonfocus_count)
        indices = [*focus, *nonfocus]
        if not indices:
            raise ValueError("trajectory BC stratified sampler produced no episode indices")
        self.rng.shuffle(indices)
        self.last_focus_episode_count = len(focus)
        self.last_nonfocus_episode_count = len(nonfocus)
        return [int(index) for index in indices]

    def _next_grouped_episode_indices(self) -> list[int]:
        batch_size = int(self.batch_episodes)
        if batch_size <= 0:
            raise ValueError("trajectory BC batch_episodes must be >= 1")
        target_focus_count = min(batch_size, int(np.ceil(batch_size * float(self.focus_fraction))))
        nonfocus_available = 0 if self.nonfocus_indices is None else int(self.nonfocus_indices.shape[0])
        if nonfocus_available <= 0:
            target_focus_count = batch_size
        group_counts = _focus_group_counts(
            batch_size=batch_size,
            target_focus_count=target_focus_count,
            fractions=tuple(float(group.fraction) for group in self.focus_groups),
        )
        focus: list[int] = []
        for group, count in zip(self.focus_groups, group_counts, strict=True):
            taken = self._take_focus_group_episodes(group, int(count))
            group.last_episode_count = len(taken)
            focus.extend(taken)
        nonfocus_count = batch_size - len(focus)
        nonfocus = self._take_nonfocus_episodes(nonfocus_count)
        indices = [*focus, *nonfocus]
        if not indices:
            raise ValueError("trajectory BC grouped sampler produced no episode indices")
        self.rng.shuffle(indices)
        self.last_focus_episode_count = len(focus)
        self.last_nonfocus_episode_count = len(nonfocus)
        return [int(index) for index in indices]

    def _take_focus_group_episodes(self, group: TrajectoryBcReplayFocusGroupState, count: int) -> list[int]:
        if count <= 0:
            return []
        taken, group.order, group.cursor = _take_from_order(
            order=group.order,
            cursor=group.cursor,
            count=int(count),
            source_indices=group.indices,
            rng=self.rng,
        )
        return taken

    def _take_focus_episodes(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if self.focus_indices is None or int(self.focus_indices.shape[0]) <= 0:
            return []
        if self.focus_order is None or int(self.focus_order.shape[0]) <= 0:
            self.focus_order = self.rng.permutation(self.focus_indices)
            self.focus_cursor = 0
        taken, self.focus_order, self.focus_cursor = _take_from_order(
            order=self.focus_order,
            cursor=self.focus_cursor,
            count=int(count),
            source_indices=self.focus_indices,
            rng=self.rng,
        )
        return taken

    def _take_nonfocus_episodes(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if self.nonfocus_indices is None or int(self.nonfocus_indices.shape[0]) <= 0:
            return []
        if self.nonfocus_order is None or int(self.nonfocus_order.shape[0]) <= 0:
            self.nonfocus_order = self.rng.permutation(self.nonfocus_indices)
            self.nonfocus_cursor = 0
        taken, self.nonfocus_order, self.nonfocus_cursor = _take_from_order(
            order=self.nonfocus_order,
            cursor=self.nonfocus_cursor,
            count=int(count),
            source_indices=self.nonfocus_indices,
            rng=self.rng,
        )
        return taken


def maybe_run_trajectory_bc_replay(
    *,
    state: TrajectoryBcReplayState | None,
    learner: Any,
    training_config: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured replay-BC auxiliary steps after an RL update."""

    if state is None:
        return
    if int(update_count) <= 0 or int(update_count) % int(state.every_updates) != 0:
        return
    structured_aux = training_config.structured_aux
    previous = _capture_teacher_state(learner)
    learner.set_teacher_aux_coefs(
        family=float(getattr(structured_aux, "trajectory_bc_teacher_family_coef", 0.05)),
        slot=float(getattr(structured_aux, "trajectory_bc_teacher_slot_coef", 0.05)),
        move_source=float(getattr(structured_aux, "trajectory_bc_teacher_move_source_coef", 0.02)),
        attack_type=float(getattr(structured_aux, "trajectory_bc_teacher_attack_type_coef", 0.02)),
        action=float(getattr(structured_aux, "trajectory_bc_teacher_action_coef", 0.20)),
        same_family_action=float(getattr(structured_aux, "trajectory_bc_teacher_same_family_action_coef", 0.60)),
        action_margin=0.0,
        same_family_action_margin=float(
            getattr(structured_aux, "trajectory_bc_teacher_same_family_action_margin_coef", 0.10)
        ),
        same_family_action_margin_value=float(
            getattr(structured_aux, "trajectory_bc_teacher_same_family_action_margin", 0.5)
        ),
        public_heuristic=0.0,
        public_nonpass_over_pass=0.0,
        exact_action_families=(),
    )
    learner.teacher_aux_mode = "warmstart_only"
    try:
        aux_metrics: dict[str, float] = {}
        total_batch_episodes = 0
        total_focus_episodes = 0
        total_nonfocus_episodes = 0
        total_focus_group_counts = {group.name: 0 for group in state.focus_groups}
        for _ in range(int(state.aux_updates)):
            indices = state.next_episode_indices()
            total_batch_episodes += len(indices)
            total_focus_episodes += int(state.last_focus_episode_count)
            total_nonfocus_episodes += int(state.last_nonfocus_episode_count)
            for group in state.focus_groups:
                total_focus_group_counts[group.name] = total_focus_group_counts.get(group.name, 0) + int(
                    group.last_episode_count
                )
            hidden = _initial_hidden_state(learner.model, batch_size=len(indices), device=device)
            batch = replay_trajectory_bc_batch(
                state.dataset,
                episode_indices=indices,
                initial_hidden_state=hidden,
            )
            aux_metrics = learner.auxiliary_update(batch)
        latest_metrics["trajectory_bc_replay_aux_updates"] = float(state.aux_updates)
        latest_metrics["trajectory_bc_replay_batch_episodes"] = float(total_batch_episodes)
        latest_metrics["trajectory_bc_replay_dataset_train_rows"] = float(state.dataset.metadata["train_rows"])
        latest_metrics["trajectory_bc_replay_focus_fraction"] = float(state.focus_fraction)
        latest_metrics["trajectory_bc_replay_focus_batch_episodes"] = float(total_focus_episodes)
        latest_metrics["trajectory_bc_replay_nonfocus_batch_episodes"] = float(total_nonfocus_episodes)
        if state.focus_groups:
            latest_metrics["trajectory_bc_replay_focus_group_count"] = float(len(state.focus_groups))
            for group in state.focus_groups:
                key = _metric_key_fragment(group.name)
                latest_metrics[f"trajectory_bc_replay_focus_group_{key}_batch_episodes"] = float(
                    total_focus_group_counts.get(group.name, 0)
                )
        for key, value in aux_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                latest_metrics[f"trajectory_bc_replay_{key}"] = float(value)
    finally:
        _restore_teacher_state(learner, previous)


def _initial_hidden_state(model: Any, *, batch_size: int, device: torch.device) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
    return model.initial_seat_hidden(int(batch_size), device=device).detach().cpu().numpy()


def _episode_indices_by_source_label(
    dataset: ReplayTrajectoryDataset,
    *,
    source_labels: tuple[str, ...],
    dataset_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not source_labels:
        return None, None
    labels = _source_labels_by_episode(dataset)
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


def _focus_group_configs_from_structured_aux(structured_aux: Any) -> tuple[Any, ...]:
    raw_groups = getattr(structured_aux, "trajectory_bc_focus_groups", ())
    if raw_groups is None:
        return ()
    return tuple(raw_groups)


def _focus_groups_by_source_label(
    dataset: ReplayTrajectoryDataset,
    *,
    focus_groups: tuple[Any, ...],
    dataset_path: Path,
    rng: np.random.Generator,
) -> tuple[tuple[TrajectoryBcReplayFocusGroupState, ...], np.ndarray | None]:
    if not focus_groups:
        return (), None
    labels = _source_labels_by_episode(dataset)
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


def _focus_group_counts(*, batch_size: int, target_focus_count: int, fractions: tuple[float, ...]) -> tuple[int, ...]:
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


def _source_labels_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return ["" for _ in range(int(dataset.episode_count))]
    labels: list[str] = []
    for bundle in bundles:
        label = bundle.get("source_dataset_label") if isinstance(bundle, dict) else None
        labels.append(str(label or ""))
    return labels


def _take_from_order(
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


def _metric_key_fragment(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "group"


def _capture_teacher_state(learner: Any) -> dict[str, Any]:
    return {
        "teacher_aux_mode": str(getattr(learner, "teacher_aux_mode", "always")),
        "teacher_family_coef": float(getattr(learner, "teacher_family_coef", 0.0)),
        "teacher_slot_coef": float(getattr(learner, "teacher_slot_coef", 0.0)),
        "teacher_hand_coef": float(getattr(learner, "teacher_hand_coef", 0.0)),
        "teacher_move_source_coef": float(getattr(learner, "teacher_move_source_coef", 0.0)),
        "teacher_attack_type_coef": float(getattr(learner, "teacher_attack_type_coef", 0.0)),
        "teacher_action_coef": float(getattr(learner, "teacher_action_coef", 0.0)),
        "teacher_same_family_action_coef": float(getattr(learner, "teacher_same_family_action_coef", 0.0)),
        "teacher_action_margin_coef": float(getattr(learner, "teacher_action_margin_coef", 0.0)),
        "teacher_action_margin": float(getattr(learner, "teacher_action_margin", 0.5)),
        "teacher_same_family_action_margin_coef": float(
            getattr(learner, "teacher_same_family_action_margin_coef", 0.0)
        ),
        "teacher_same_family_action_margin": float(getattr(learner, "teacher_same_family_action_margin", 0.5)),
        "teacher_exact_action_families": tuple(getattr(learner, "teacher_exact_action_families", ())),
        "teacher_public_heuristic_coef": float(getattr(learner, "teacher_public_heuristic_coef", 0.0)),
        "teacher_public_heuristic_temperature": float(getattr(learner, "teacher_public_heuristic_temperature", 32.0)),
        "teacher_public_nonpass_over_pass_coef": float(getattr(learner, "teacher_public_nonpass_over_pass_coef", 0.0)),
        "teacher_public_nonpass_over_pass_margin": float(
            getattr(learner, "teacher_public_nonpass_over_pass_margin", 0.5)
        ),
        "teacher_public_heuristic_families": tuple(getattr(learner, "teacher_public_heuristic_families", ())),
        "teacher_public_heuristic_profiles": tuple(getattr(learner, "teacher_public_heuristic_profiles", ())),
        "teacher_public_heuristic_profile_mode": str(getattr(learner, "teacher_public_heuristic_profile_mode", "")),
        "teacher_public_heuristic_profiles_end_updates": int(
            getattr(learner, "teacher_public_heuristic_profiles_end_updates", -1)
        ),
    }


def _restore_teacher_state(learner: Any, state: dict[str, Any]) -> None:
    learner.set_teacher_aux_coefs(
        family=state["teacher_family_coef"],
        slot=state["teacher_slot_coef"],
        hand=state["teacher_hand_coef"],
        move_source=state["teacher_move_source_coef"],
        attack_type=state["teacher_attack_type_coef"],
        action=state["teacher_action_coef"],
        same_family_action=state["teacher_same_family_action_coef"],
        action_margin=state["teacher_action_margin_coef"],
        action_margin_value=state["teacher_action_margin"],
        same_family_action_margin=state["teacher_same_family_action_margin_coef"],
        same_family_action_margin_value=state["teacher_same_family_action_margin"],
        exact_action_families=state["teacher_exact_action_families"],
        public_heuristic=state["teacher_public_heuristic_coef"],
        public_heuristic_temperature=state["teacher_public_heuristic_temperature"],
        public_nonpass_over_pass=state["teacher_public_nonpass_over_pass_coef"],
        public_nonpass_over_pass_margin=state["teacher_public_nonpass_over_pass_margin"],
        public_heuristic_families=state["teacher_public_heuristic_families"],
        public_heuristic_profiles=state["teacher_public_heuristic_profiles"],
        public_heuristic_profile_mode=state["teacher_public_heuristic_profile_mode"],
        public_heuristic_profiles_end_updates=state["teacher_public_heuristic_profiles_end_updates"],
    )
    learner.teacher_aux_mode = state["teacher_aux_mode"]


__all__ = [
    "TrajectoryBcReplayState",
    "maybe_run_trajectory_bc_replay",
]
