"""Replay trajectory extraction for targeted behavior-cloning warmstarts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config import StackConfig, load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.models.observation_contract import header_field_index
from weiss_rl.replay.bundles import ReplayStep, load_replay_bundle
from weiss_rl.replay.inspector import (
    _legal_ids_for_env_row,
    _load_action_catalog,
    _load_observation_layout,
    _load_run_spec_bundle,
    _pass_action_id,
    _require_initial_identity,
    _require_post_step_match,
    _require_pre_step_match,
    _require_single_env_batch,
)
from weiss_rl.replay.runner import ReplayEnvFactory, build_replay_env, require_supported_rerun_contract
from weiss_rl.replay.trajectory_bc_dataset import (
    BC_DATASET_FORMAT,
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    merge_replay_trajectory_bc_datasets,
    replay_trajectory_bc_batch,
    save_replay_trajectory_bc_dataset,
    subset_replay_trajectory_bc_dataset,
)
from weiss_rl.runtime.components.action_surface import (
    filter_batch_main_move_only_rows_to_pass,
    filter_batch_mulligan_select_after_select,
    filter_batch_pass_when_attack_available,
)
from weiss_rl.runtime.components.legal_meta import action_catalog_indices, legal_action_meta_from_ids
from weiss_rl.runtime.components.teacher_labels import teacher_labels_from_actions

_PAIR_SWAP_RE = re.compile(r"_pair(?P<pair>\d+)_swap(?P<swap>\d+)")


@dataclass(frozen=True, slots=True)
class _BundleSelection:
    bundle_path: Path
    pair_index: int | None
    swap_index: int | None
    focal_seat: int
    outcome: str | None
    episode_seed: int | None


@dataclass(slots=True)
class _StepRow:
    obs: np.ndarray
    actor: int
    action: int
    legal_ids: np.ndarray
    legal_action_meta: np.ndarray
    teacher_family: int
    teacher_slot: int
    teacher_move_source: int
    teacher_attack_type: int
    teacher_action: int
    teacher_valid: bool
    policy_train: bool
    teacher_action_overridden: bool
    decision_kind: int
    supported_target: bool


def build_replay_trajectory_bc_dataset(
    *,
    bundle_paths: Sequence[Path],
    run_dir: Path,
    stack: StackConfig | Path,
    episodes_jsonl: Path | None = None,
    include_outcomes: Iterable[str] = ("W",),
    focal_seat: int | None = None,
    max_bundles: int | None = None,
    teacher_action_overrides: Mapping[tuple[str, int], int] | None = None,
    env_factory: ReplayEnvFactory | None = None,
) -> ReplayTrajectoryDataset:
    """Rerun replay bundles and return focal-seat recorded-action supervision.

    The extractor keeps full episode sequences so recurrent hidden state is trained
    along the same order as the replay. Only supported focal rows are marked as
    trainable; opponent rows and padding rows remain in the sequence with zero
    loss so the recurrent scan stays well shaped.
    """

    bundle_list = [Path(path).resolve() for path in bundle_paths]
    if not bundle_list:
        raise ValueError("bundle_paths must contain at least one replay bundle")
    if max_bundles is not None:
        limit = int(max_bundles)
        if limit <= 0:
            raise ValueError("max_bundles must be positive when provided")
        bundle_list = bundle_list[:limit]

    stack_config = load_stack_config(stack) if isinstance(stack, Path) else stack
    run_spec_bundle = _load_run_spec_bundle(Path(run_dir).resolve())
    if run_spec_bundle is None:
        raise FileNotFoundError(f"spec_bundle.json not found in run_dir: {Path(run_dir).resolve()}")
    action_catalog = _load_action_catalog(run_spec_bundle)
    if action_catalog is None:
        raise RuntimeError("Replay trajectory BC extraction requires a structured action catalog")
    family_index, attack_type_index = action_catalog_indices(action_catalog)
    pass_action_id = _pass_action_id(run_spec_bundle)
    episode_records = _load_episode_records(episodes_jsonl)
    selections = _select_bundles(
        bundle_list,
        episode_records=episode_records,
        include_outcomes=tuple(include_outcomes),
        focal_seat=focal_seat,
    )
    if not selections:
        raise ValueError("No replay bundles matched the requested outcome/focal-seat filters")
    override_map = dict(teacher_action_overrides or {})

    episodes: list[list[_StepRow]] = []
    summary_counter: Counter[str] = Counter()
    selected_metadata: list[dict[str, Any]] = []
    for selection in selections:
        rows, row_counts = _extract_bundle_rows(
            selection=selection,
            stack=stack_config,
            run_spec_bundle=run_spec_bundle,
            action_catalog=action_catalog,
            family_index=family_index,
            attack_type_index=attack_type_index,
            pass_action_id=pass_action_id,
            teacher_action_overrides=override_map,
            env_factory=env_factory,
        )
        if not rows:
            continue
        episodes.append(rows)
        summary_counter.update(row_counts)
        selected_metadata.append(
            {
                "bundle_path": selection.bundle_path.as_posix(),
                "pair_index": selection.pair_index,
                "swap_index": selection.swap_index,
                "focal_seat": selection.focal_seat,
                "outcome": selection.outcome,
                "episode_seed": selection.episode_seed,
                "steps": len(rows),
                "train_rows": int(row_counts["train_rows"]),
                "teacher_action_override_rows": int(row_counts["teacher_action_override_rows"]),
                "nonoverride_focal_rows": int(row_counts["nonoverride_focal_rows"]),
                "unsupported_target_rows": int(row_counts["unsupported_target_rows"]),
            }
        )

    if not episodes:
        raise ValueError("Selected replay bundles did not produce any trajectory rows")

    dataset = _collate_episodes(
        episodes,
        pass_action_id=pass_action_id,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
    metadata = {
        "format": BC_DATASET_FORMAT,
        "run_dir": Path(run_dir).resolve().as_posix(),
        "bundle_count": len(episodes),
        "requested_bundle_count": len(bundle_list),
        "include_outcomes": [str(value) for value in include_outcomes],
        "obs_dim": int(dataset["obs"].shape[-1]),
        "time_steps": int(dataset["obs"].shape[0]),
        "episode_count": int(dataset["obs"].shape[1]),
        "row_count": int(dataset["obs"].shape[0] * dataset["obs"].shape[1]),
        "train_rows": int(np.count_nonzero(dataset["policy_train_mask"])),
        "teacher_valid_rows": int(np.count_nonzero(dataset["teacher_valid"])),
        "supported_target_rows": int(summary_counter["supported_target_rows"]),
        "teacher_action_override_rows": int(summary_counter["teacher_action_override_rows"]),
        "teacher_action_override_key_count": len(override_map),
        "nonoverride_focal_rows": int(summary_counter["nonoverride_focal_rows"]),
        "unsupported_target_rows": int(summary_counter["unsupported_target_rows"]),
        "opponent_rows": int(summary_counter["opponent_rows"]),
        "nonfocal_rows": int(summary_counter["nonfocal_rows"]),
        "pass_action_id": int(pass_action_id),
        "spec_hash256": _first_spec_hash(selections),
        "selected_bundles": selected_metadata,
    }
    return ReplayTrajectoryDataset(metadata=metadata, **dataset)


def load_teacher_action_overrides_jsonl(path: Path) -> dict[tuple[str, int], int]:
    """Load bundle/step teacher-action overrides from a JSONL manifest."""

    overrides: dict[tuple[str, int], int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"teacher-action override rows must be objects: {path}:{line_number}")
            raw_bundle_path = payload.get("bundle_path")
            raw_bundle_name = payload.get("bundle_name")
            if not isinstance(raw_bundle_path, str) and not isinstance(raw_bundle_name, str):
                raise ValueError(f"override row must include bundle_path or bundle_name: {path}:{line_number}")
            step_index = int(payload["step_index"])
            teacher_action = int(payload["teacher_action"])
            keys: list[tuple[str, int]] = []
            if isinstance(raw_bundle_path, str) and raw_bundle_path:
                bundle_path = Path(raw_bundle_path)
                keys.append((bundle_path.resolve().as_posix(), step_index))
                keys.append((bundle_path.name, step_index))
            if isinstance(raw_bundle_name, str) and raw_bundle_name:
                keys.append((raw_bundle_name, step_index))
            for key in keys:
                previous = overrides.get(key)
                if previous is not None and int(previous) != teacher_action:
                    raise ValueError(f"conflicting teacher-action override for {key}: {previous} vs {teacher_action}")
                overrides[key] = teacher_action
    if not overrides:
        raise ValueError(f"no teacher-action overrides found in {path}")
    return overrides


def _extract_bundle_rows(
    *,
    selection: _BundleSelection,
    stack: StackConfig,
    run_spec_bundle: Mapping[str, Any],
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    pass_action_id: int,
    teacher_action_overrides: Mapping[tuple[str, int], int],
    env_factory: ReplayEnvFactory | None,
) -> tuple[list[_StepRow], Counter[str]]:
    meta, steps, _fault = load_replay_bundle(selection.bundle_path)
    contract = require_supported_rerun_contract(meta)
    env = None
    rows: list[_StepRow] = []
    counts: Counter[str] = Counter()
    try:
        env = build_replay_env(contract, env_factory=env_factory)
        current_batch = _require_single_env_batch(env.reset(seed=meta.episode_seed64), context="reset")
        _require_initial_identity(meta=meta, batch=current_batch)
        spec_hash256 = bytes.fromhex(meta.spec_hash256)
        for step_index, expected_step in enumerate(steps):
            _require_pre_step_match(
                step_index=step_index,
                expected_step=expected_step,
                current_batch=current_batch,
                spec_hash256=spec_hash256,
            )
            current_batch = _batch_with_legal_meta(
                current_batch,
                action_catalog=action_catalog,
                family_index=family_index,
                attack_type_index=attack_type_index,
            )
            raw_legal_ids = _legal_ids_for_env_row(current_batch)
            filtered_batch, legal_ids = _filter_training_action_surface(
                batch=current_batch,
                legal_ids=raw_legal_ids,
                stack=stack,
                action_catalog=action_catalog,
                run_spec_bundle=run_spec_bundle,
                pass_action_id=pass_action_id,
            )
            row = _build_step_row(
                batch=filtered_batch,
                expected_step=expected_step,
                focal_seat=selection.focal_seat,
                legal_ids=legal_ids,
                action_catalog=action_catalog,
                family_index=family_index,
                attack_type_index=attack_type_index,
                teacher_action_override=_teacher_action_override_for(
                    teacher_action_overrides,
                    bundle_path=selection.bundle_path,
                    step_index=step_index,
                ),
                override_mode=bool(teacher_action_overrides),
            )
            rows.append(row)
            counts["rows"] += 1
            if row.actor == selection.focal_seat:
                counts["focal_rows"] += 1
            else:
                counts["opponent_rows"] += 1
                counts["nonfocal_rows"] += 1
            if row.supported_target:
                counts["supported_target_rows"] += 1
            if row.teacher_action_overridden:
                counts["teacher_action_override_rows"] += 1
            if row.policy_train:
                counts["train_rows"] += 1
            elif row.actor == selection.focal_seat and (
                not bool(teacher_action_overrides) or row.teacher_action_overridden
            ):
                counts["unsupported_target_rows"] += 1
            elif row.actor == selection.focal_seat:
                counts["nonoverride_focal_rows"] += 1

            next_batch = _require_single_env_batch(
                env.step(np.asarray([expected_step.action], dtype=np.uint32)),
                context=f"step[{step_index}]",
            )
            _require_post_step_match(step_index=step_index, expected_step=expected_step, next_batch=next_batch)
            current_batch = next_batch
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return rows, counts


def _build_step_row(
    *,
    batch: DecisionBoundaryBatch,
    expected_step: ReplayStep,
    focal_seat: int,
    legal_ids: np.ndarray,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    teacher_action_override: int | None,
    override_mode: bool,
) -> _StepRow:
    actor = int(batch.actor[0])
    action = int(expected_step.action)
    teacher_action = action if teacher_action_override is None else int(teacher_action_override)
    legal_ids_array = np.asarray(legal_ids, dtype=np.uint32)
    legal_meta = _single_row_legal_meta(batch, expected_count=int(legal_ids_array.shape[0]))
    supported_target = bool(np.any(legal_ids_array.astype(np.int64, copy=False) == teacher_action))
    policy_train = bool(
        actor == int(focal_seat)
        and supported_target
        and (not bool(override_mode) or teacher_action_override is not None)
    )
    labels = teacher_labels_from_actions(
        row_indices=np.asarray([0], dtype=np.int64),
        chosen_actions=np.asarray([teacher_action], dtype=np.int64),
        num_rows=1,
        guidance_active=policy_train,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, label_action, teacher_valid = labels
    return _StepRow(
        obs=np.asarray(batch.obs[0], dtype=np.float32),
        actor=actor,
        action=action,
        legal_ids=legal_ids_array,
        legal_action_meta=legal_meta,
        teacher_family=int(teacher_family[0]),
        teacher_slot=int(teacher_slot[0]),
        teacher_move_source=int(teacher_move_source[0]),
        teacher_attack_type=int(teacher_attack_type[0]),
        teacher_action=int(label_action[0]),
        teacher_valid=bool(teacher_valid[0]),
        policy_train=policy_train,
        teacher_action_overridden=teacher_action_override is not None,
        decision_kind=int(np.asarray(batch.decision_kind, dtype=np.int32)[0]),
        supported_target=supported_target,
    )


def _collate_episodes(
    episodes: Sequence[Sequence[_StepRow]],
    *,
    pass_action_id: int,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> dict[str, np.ndarray]:
    time_steps = max(len(episode) for episode in episodes)
    episode_count = len(episodes)
    obs_dim = int(episodes[0][0].obs.shape[0])
    obs = np.zeros((time_steps, episode_count, obs_dim), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int8)
    actions = np.zeros((time_steps, episode_count), dtype=np.int64)
    teacher_family = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_slot = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_move_source = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_attack_type = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_action = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_valid = np.zeros((time_steps, episode_count), dtype=np.bool_)
    policy_train_mask = np.zeros((time_steps, episode_count), dtype=np.bool_)
    reset_before_step = np.zeros((time_steps, episode_count), dtype=np.bool_)

    padding_ids, padding_meta = _padding_legal_row(
        pass_action_id=pass_action_id,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    legal_offsets = [0]
    cursor = 0
    for step_index in range(time_steps):
        for episode_index, episode in enumerate(episodes):
            if step_index < len(episode):
                row = episode[step_index]
                obs[step_index, episode_index] = row.obs
                actor[step_index, episode_index] = np.int8(row.actor)
                actions[step_index, episode_index] = int(row.action)
                teacher_family[step_index, episode_index] = row.teacher_family
                teacher_slot[step_index, episode_index] = row.teacher_slot
                teacher_move_source[step_index, episode_index] = row.teacher_move_source
                teacher_attack_type[step_index, episode_index] = row.teacher_attack_type
                teacher_action[step_index, episode_index] = row.teacher_action
                teacher_valid[step_index, episode_index] = row.teacher_valid
                policy_train_mask[step_index, episode_index] = row.policy_train
                row_ids = row.legal_ids
                row_meta = row.legal_action_meta
            else:
                row_ids = padding_ids
                row_meta = padding_meta
            legal_ids_parts.append(row_ids)
            legal_meta_parts.append(row_meta)
            cursor += int(row_ids.shape[0])
            legal_offsets.append(cursor)

    legal_ids = np.concatenate(legal_ids_parts, axis=0).astype(np.uint32, copy=False)
    legal_action_meta = np.concatenate(legal_meta_parts, axis=0).astype(np.uint16, copy=False)
    return {
        "obs": obs,
        "actor": actor,
        "to_play_seat": actor.astype(np.int8, copy=True),
        "actions": actions,
        "legal_ids": legal_ids,
        "legal_offsets": np.asarray(legal_offsets, dtype=np.uint32),
        "legal_action_meta": legal_action_meta,
        "teacher_family": teacher_family,
        "teacher_slot": teacher_slot,
        "teacher_move_source": teacher_move_source,
        "teacher_attack_type": teacher_attack_type,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "policy_train_mask": policy_train_mask,
        "reset_before_step": reset_before_step,
    }


def _batch_with_legal_meta(
    batch: DecisionBoundaryBatch,
    *,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> DecisionBoundaryBatch:
    if batch.ids_offsets is None:
        return batch
    legal_ids, _legal_offsets = batch.ids_offsets
    if batch.legal_action_meta is not None:
        return batch
    meta = legal_action_meta_from_ids(
        np.asarray(legal_ids, dtype=np.uint32),
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=3,
    )
    return replace(batch, legal_action_meta=meta)


def _filter_training_action_surface(
    *,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    stack: StackConfig,
    action_catalog: ActionCatalog,
    run_spec_bundle: Mapping[str, Any],
    pass_action_id: int,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    training_config = stack.config.training
    if training_config is None:
        return batch, legal_ids
    mulligan_guard = bool(getattr(training_config, "mulligan_force_confirm_after_select", False))
    main_move_guard = bool(getattr(training_config, "force_pass_over_main_move_only", False))
    attack_guard = bool(getattr(training_config, "force_attack_over_pass_when_attack_legal", False))
    if not mulligan_guard and not main_move_guard and not attack_guard:
        return batch, legal_ids

    filtered_batch = batch
    layout = _load_observation_layout(run_spec_bundle)
    field_index = None if layout is None else header_field_index(layout, "last_action_arg0")
    last_action_arg0_index = -1 if field_index is None else int(field_index)
    family_index, _attack_type_index = action_catalog_indices(action_catalog)
    if mulligan_guard:
        filtered_batch, _result = filter_batch_mulligan_select_after_select(
            filtered_batch,
            last_action_arg0_index=last_action_arg0_index,
            mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
            mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
        )
    if main_move_guard:
        filtered_batch, _result = filter_batch_main_move_only_rows_to_pass(
            filtered_batch,
            pass_action_id=int(pass_action_id),
            main_move_family_id=int(family_index.get("main_move", -1)),
        )
    if attack_guard:
        filtered_batch, _result = filter_batch_pass_when_attack_available(
            filtered_batch,
            pass_action_id=int(pass_action_id),
            attack_family_id=int(family_index.get("attack", -1)),
        )
    if filtered_batch.ids_offsets is None:
        return batch, legal_ids
    filtered_ids, filtered_offsets = filtered_batch.ids_offsets
    return filtered_batch, np.asarray(
        filtered_ids[int(filtered_offsets[0]) : int(filtered_offsets[1])],
        dtype=np.uint32,
    )


def _single_row_legal_meta(batch: DecisionBoundaryBatch, *, expected_count: int) -> np.ndarray:
    if batch.legal_action_meta is None:
        raise RuntimeError("Replay trajectory BC extraction requires legal action metadata")
    meta = np.asarray(batch.legal_action_meta, dtype=np.uint16)
    if meta.ndim != 2:
        raise RuntimeError("legal_action_meta must be a matrix")
    if meta.shape[0] != int(expected_count):
        raise RuntimeError(
            f"legal_action_meta row count must match filtered legal ids: expected {expected_count}, got {meta.shape[0]}"
        )
    return meta


def _padding_legal_row(
    *,
    pass_action_id: int,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([int(pass_action_id)], dtype=np.uint32)
    meta = legal_action_meta_from_ids(
        ids,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=3,
    )
    if meta is None:
        raise RuntimeError("Could not build padding legal-action metadata")
    return ids, meta


def _select_bundles(
    bundle_paths: Sequence[Path],
    *,
    episode_records: Mapping[tuple[int, int], Mapping[str, Any]],
    include_outcomes: tuple[str, ...],
    focal_seat: int | None,
) -> list[_BundleSelection]:
    allowed_outcomes = {str(item).strip().upper() for item in include_outcomes if str(item).strip()}
    selections: list[_BundleSelection] = []
    for bundle_path in bundle_paths:
        pair_swap = _pair_swap_from_bundle_path(bundle_path)
        record = None if pair_swap is None else episode_records.get(pair_swap)
        outcome = None if record is None else str(record.get("outcome", "")).strip().upper()
        if allowed_outcomes and outcome and outcome not in allowed_outcomes:
            continue
        resolved_focal_seat = focal_seat
        if resolved_focal_seat is None and record is not None:
            resolved_focal_seat = int(record["focal_seat"])
        if resolved_focal_seat is None:
            raise ValueError(
                f"Could not infer focal seat for {bundle_path}; pass episodes_jsonl or explicit focal_seat"
            )
        selections.append(
            _BundleSelection(
                bundle_path=bundle_path,
                pair_index=None if pair_swap is None else pair_swap[0],
                swap_index=None if pair_swap is None else pair_swap[1],
                focal_seat=int(resolved_focal_seat),
                outcome=outcome or None,
                episode_seed=None if record is None else int(record.get("episode_seed", 0)),
            )
        )
    return selections


def _load_episode_records(path: Path | None) -> dict[tuple[int, int], Mapping[str, Any]]:
    if path is None:
        return {}
    records: dict[tuple[int, int], Mapping[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"episodes_jsonl rows must be objects: {path}")
            key = (int(payload["pair_index"]), int(payload["swap_index"]))
            records[key] = payload
    return records


def _teacher_action_override_for(
    overrides: Mapping[tuple[str, int], int],
    *,
    bundle_path: Path,
    step_index: int,
) -> int | None:
    if not overrides:
        return None
    path = Path(bundle_path)
    keys = (
        (path.resolve().as_posix(), int(step_index)),
        (path.name, int(step_index)),
    )
    for key in keys:
        if key in overrides:
            return int(overrides[key])
    return None


def _pair_swap_from_bundle_path(path: Path) -> tuple[int, int] | None:
    match = _PAIR_SWAP_RE.search(Path(path).stem)
    if match is None:
        return None
    return int(match.group("pair")), int(match.group("swap"))


def _first_spec_hash(selections: Sequence[_BundleSelection]) -> str | None:
    for selection in selections:
        meta, _steps, _fault = load_replay_bundle(selection.bundle_path)
        return str(meta.spec_hash256)
    return None


__all__ = [
    "BC_DATASET_FORMAT",
    "ReplayTrajectoryDataset",
    "build_replay_trajectory_bc_dataset",
    "load_teacher_action_overrides_jsonl",
    "load_replay_trajectory_bc_dataset",
    "merge_replay_trajectory_bc_datasets",
    "replay_trajectory_bc_batch",
    "save_replay_trajectory_bc_dataset",
    "subset_replay_trajectory_bc_dataset",
]
