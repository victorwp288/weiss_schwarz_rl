"""Replay trajectory extraction for targeted behavior-cloning warmstarts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config import StackConfig, load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.replay.bundles import load_replay_bundle
from weiss_rl.replay.inspection_policy_loading import load_action_catalog, load_run_spec_bundle
from weiss_rl.replay.rerun_validation import (
    legal_ids_for_env_row,
    pass_action_id_from_spec_bundle,
    require_initial_identity,
    require_post_step_match,
    require_pre_step_match,
    require_single_env_batch,
)
from weiss_rl.replay.runner import ReplayEnvFactory, build_replay_env, require_supported_rerun_contract
from weiss_rl.replay.trajectory_bc_batching import replay_trajectory_bc_batch, subset_replay_trajectory_bc_dataset
from weiss_rl.replay.trajectory_bc_dataset import merge_replay_trajectory_bc_datasets
from weiss_rl.replay.trajectory_bc_dataset_schema import (
    BC_DATASET_FORMAT,
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    save_replay_trajectory_bc_dataset,
)
from weiss_rl.replay.trajectory_bc_rows import (
    TrajectoryBcStepRow,
    batch_with_legal_meta,
    build_step_row,
    collate_episode_rows,
    filter_training_action_surface,
)
from weiss_rl.replay.trajectory_bc_selection import (
    BundleSelection,
    first_spec_hash,
    load_episode_records,
    load_teacher_action_overrides_jsonl,
    select_bundles,
    teacher_action_override_for,
)
from weiss_rl.runtime.components.actions.legal_meta import action_catalog_indices


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
    run_spec_bundle = load_run_spec_bundle(Path(run_dir).resolve())
    if run_spec_bundle is None:
        raise FileNotFoundError(f"spec_bundle.json not found in run_dir: {Path(run_dir).resolve()}")
    action_catalog = load_action_catalog(run_spec_bundle)
    if action_catalog is None:
        raise RuntimeError("Replay trajectory BC extraction requires a structured action catalog")
    family_index, attack_type_index = action_catalog_indices(action_catalog)
    pass_action_id = pass_action_id_from_spec_bundle(run_spec_bundle)
    episode_records = load_episode_records(episodes_jsonl)
    selections = select_bundles(
        bundle_list,
        episode_records=episode_records,
        include_outcomes=tuple(include_outcomes),
        focal_seat=focal_seat,
    )
    if not selections:
        raise ValueError("No replay bundles matched the requested outcome/focal-seat filters")
    override_map = dict(teacher_action_overrides or {})

    episodes: list[list[TrajectoryBcStepRow]] = []
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

    dataset = collate_episode_rows(
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
        "spec_hash256": first_spec_hash(selections),
        "selected_bundles": selected_metadata,
    }
    return ReplayTrajectoryDataset(metadata=metadata, **dataset)


def _extract_bundle_rows(
    *,
    selection: BundleSelection,
    stack: StackConfig,
    run_spec_bundle: Mapping[str, Any],
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    pass_action_id: int,
    teacher_action_overrides: Mapping[tuple[str, int], int],
    env_factory: ReplayEnvFactory | None,
) -> tuple[list[TrajectoryBcStepRow], Counter[str]]:
    meta, steps, _fault = load_replay_bundle(selection.bundle_path)
    contract = require_supported_rerun_contract(meta)
    env = None
    rows: list[TrajectoryBcStepRow] = []
    counts: Counter[str] = Counter()
    try:
        env = build_replay_env(contract, env_factory=env_factory)
        current_batch = require_single_env_batch(
            env.reset(seed=meta.episode_seed64),
            context="reset",
            owner="Replay trajectory BC extraction",
        )
        require_initial_identity(meta=meta, batch=current_batch)
        spec_hash256 = bytes.fromhex(meta.spec_hash256)
        for step_index, expected_step in enumerate(steps):
            require_pre_step_match(
                step_index=step_index,
                expected_step=expected_step,
                current_batch=current_batch,
                spec_hash256=spec_hash256,
                owner="Replay trajectory BC extraction",
            )
            current_batch = batch_with_legal_meta(
                current_batch,
                action_catalog=action_catalog,
                family_index=family_index,
                attack_type_index=attack_type_index,
            )
            raw_legal_ids = legal_ids_for_env_row(current_batch, owner="Replay trajectory BC extraction")
            filtered_batch, legal_ids = filter_training_action_surface(
                batch=current_batch,
                legal_ids=raw_legal_ids,
                stack=stack,
                action_catalog=action_catalog,
                run_spec_bundle=run_spec_bundle,
                pass_action_id=pass_action_id,
            )
            row = build_step_row(
                batch=filtered_batch,
                expected_step=expected_step,
                focal_seat=selection.focal_seat,
                legal_ids=legal_ids,
                action_catalog=action_catalog,
                family_index=family_index,
                attack_type_index=attack_type_index,
                teacher_action_override=teacher_action_override_for(
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

            next_batch = require_single_env_batch(
                env.step(np.asarray([expected_step.action], dtype=np.uint32)),
                context=f"step[{step_index}]",
                owner="Replay trajectory BC extraction",
            )
            require_post_step_match(step_index=step_index, expected_step=expected_step, next_batch=next_batch)
            current_batch = next_batch
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return rows, counts


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
