"""Canonical minimal single-node training loop used by ``scripts.train``."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.runtime import QueueRuntimeMode
from weiss_rl.training.paired_outcome_preference_replay import (
    PairedOutcomePreferenceReplayState,
    maybe_run_paired_outcome_preference_replay,
)
from weiss_rl.training.paired_swing_replay import PairedSwingReplayState, maybe_run_paired_swing_replay
from weiss_rl.training.trajectory_bc_replay import TrajectoryBcReplayState, maybe_run_trajectory_bc_replay

_POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES = (
    "trajectory_bc_replay_",
    "paired_swing_replay_",
    "paired_outcome_preference_replay_",
    "pfsp_",
    "collector_pfsp_",
)


@dataclass(frozen=True, slots=True)
class MinimalTrainingHooks:
    configure_torch_threads: Any
    spec_dimensions: Any
    experiment_role: Any
    training_paths: Any
    validate_algorithm_model_contract: Any
    build_policy_value_model: Any
    maybe_compile_learner_model: Any
    build_training_learner: Any
    restore_learner_from_checkpoint: Any
    initialize_learner_from_checkpoint: Any
    compute_config_hash256: Any
    ensure_noleague_baseline_anchor: Any
    import_seed_snapshot_pool: Any
    canonical_config_dict: Any
    build_runtime_config: Any
    queue_runtime_cls: Any
    central_runtime_actor_torch_threads: Any
    build_training_profiler: Any
    run_structured_warmstart: Any
    profile_block: Any
    apply_guidance_schedule_for_next_update: Any
    entropy_coef_for_next_update: Any
    torch_num_threads_scope: Any
    collect_training_batch: Any
    write_scalars_record: Any
    write_checkpoint: Any
    publish_checkpoint_aliases: Any
    maybe_log_structured_mainmove_guard: Any
    persist_snapshot_registry_entry: Any
    is_noleague_baseline_role: Any
    run_snapshot_promotion_gate: Any
    should_run_periodic_dev_eval: Any
    run_periodic_dev_eval: Any
    slug_policy_id: Any
    load_checkpoint_tracker: Any
    confirmatory_dev_eval_request: Any
    periodic_dev_eval_schedule: Any
    expand_periodic_dev_eval_paired_seeds: Any
    ensure_current_checkpoint: Any
    maybe_rollback_to_best_checkpoint: Any
    maybe_finalize_from_best_checkpoint: Any


def _publish_initial_runtime_snapshot_after_resume(*, runtime: Any, model: Any, update_count: int) -> dict[str, float]:
    """Synchronize freshly constructed actors with a nonzero resumed learner update."""

    if int(update_count) <= 0:
        return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
    return dict(
        runtime.maybe_publish_snapshot(
            learner_model=model,
            learner_update_count=int(update_count),
            force=True,
        )
    )


def _schedule_update_count_for_next_update(
    *,
    learner_update_count: int,
    init_schedule_offset_updates: int,
) -> int:
    """Map fresh-run local updates onto source-checkpoint schedule time."""

    return max(0, int(init_schedule_offset_updates)) + max(0, int(learner_update_count)) + 1


def _effective_init_schedule_offset_from_checkpoint(
    *,
    source_update_count: int,
    source_init_schedule_offset_updates: int,
    override_updates: int | None = None,
) -> int:
    """Carry cumulative schedule time across fresh warm-start segments."""

    if override_updates is not None:
        return max(0, int(override_updates))
    return max(0, int(source_init_schedule_offset_updates)) + max(0, int(source_update_count))


def _infer_init_schedule_offset_from_scalars(scalars_path: Path) -> int:
    """Recover warm-start schedule offset for older checkpoints that did not persist it."""

    if not scalars_path.is_file():
        return 0
    try:
        lines = scalars_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        offset = record.get("init_schedule_offset_updates") if isinstance(record, dict) else None
        if offset is None:
            continue
        try:
            return max(0, int(float(offset)))
        except (TypeError, ValueError):
            continue
    return 0


def _merge_post_update_auxiliary_metrics_into_training_log(*, learner: Any, metrics: Mapping[str, float]) -> None:
    logger = getattr(learner, "logger", None)
    if logger is None:
        return
    merge_latest = getattr(logger, "merge_latest_custom_metrics", None)
    if not callable(merge_latest):
        return
    merge_latest(
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        metrics=metrics,
        prefixes=_POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    )


def _reset_policy_anchor_to_current_model(learner: Any) -> None:
    reset = getattr(learner, "reset_policy_anchor_to_current_model", None)
    if not callable(reset):
        raise ValueError("paired outcome preference replay requires learner policy-anchor support")
    try:
        reset(force=True)
    except TypeError:
        reset()


def run_minimal_training(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    profile: str,
    device: torch.device,
    seed: int,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    runtime_mode: QueueRuntimeMode,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None = None,
    profile_timers: bool = False,
    torch_profiler: bool = False,
    resume_checkpoint_path: Path | None = None,
    init_from_checkpoint_path: Path | None = None,
    init_schedule_offset_override_updates: int | None = None,
    tensorboard_logger: TensorBoardLogger | None = None,
    hooks: MinimalTrainingHooks,
) -> dict[str, float]:
    _configure_torch_threads = hooks.configure_torch_threads
    _spec_dimensions = hooks.spec_dimensions
    _experiment_role = hooks.experiment_role
    _training_paths = hooks.training_paths
    _validate_algorithm_model_contract = hooks.validate_algorithm_model_contract
    build_policy_value_model = hooks.build_policy_value_model
    _maybe_compile_learner_model = hooks.maybe_compile_learner_model
    _build_training_learner = hooks.build_training_learner
    _restore_learner_from_checkpoint = hooks.restore_learner_from_checkpoint
    _initialize_learner_from_checkpoint = hooks.initialize_learner_from_checkpoint
    compute_config_hash256 = hooks.compute_config_hash256
    _ensure_noleague_baseline_anchor = hooks.ensure_noleague_baseline_anchor
    _import_seed_snapshot_pool = hooks.import_seed_snapshot_pool
    canonical_config_dict = hooks.canonical_config_dict
    build_runtime_config = hooks.build_runtime_config
    QueueRuntime = hooks.queue_runtime_cls
    _central_runtime_actor_torch_threads = hooks.central_runtime_actor_torch_threads
    build_training_profiler = hooks.build_training_profiler
    _run_structured_warmstart = hooks.run_structured_warmstart
    profile_block = hooks.profile_block
    _apply_guidance_schedule_for_next_update = hooks.apply_guidance_schedule_for_next_update
    _entropy_coef_for_next_update = hooks.entropy_coef_for_next_update
    _torch_num_threads_scope = hooks.torch_num_threads_scope
    collect_training_batch = hooks.collect_training_batch
    _write_scalars_record = hooks.write_scalars_record
    _write_checkpoint = hooks.write_checkpoint
    _publish_checkpoint_aliases = hooks.publish_checkpoint_aliases
    _maybe_log_structured_mainmove_guard = hooks.maybe_log_structured_mainmove_guard
    _persist_snapshot_registry_entry = hooks.persist_snapshot_registry_entry
    _is_noleague_baseline_role = hooks.is_noleague_baseline_role
    _run_snapshot_promotion_gate = hooks.run_snapshot_promotion_gate
    _should_run_periodic_dev_eval = hooks.should_run_periodic_dev_eval
    _run_periodic_dev_eval = hooks.run_periodic_dev_eval
    _slug_policy_id = hooks.slug_policy_id
    _load_checkpoint_tracker = hooks.load_checkpoint_tracker
    _confirmatory_dev_eval_request = hooks.confirmatory_dev_eval_request
    _periodic_dev_eval_schedule = hooks.periodic_dev_eval_schedule
    _expand_periodic_dev_eval_paired_seeds = hooks.expand_periodic_dev_eval_paired_seeds
    _ensure_current_checkpoint = hooks.ensure_current_checkpoint
    _maybe_rollback_to_best_checkpoint = hooks.maybe_rollback_to_best_checkpoint
    _maybe_finalize_from_best_checkpoint = hooks.maybe_finalize_from_best_checkpoint
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    observation_dim, action_dim = _spec_dimensions(contract)
    training_config = stack.config.training
    model_config = stack.config.model
    environment_config = stack.config.environment
    rewards_config = stack.config.rewards
    if training_config is None or model_config is None or environment_config is None or rewards_config is None:
        raise RuntimeError("The locked stack is missing training, model, environment, or rewards config")

    training_paths = _training_paths(artifacts.run_dir)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    algorithm = str(training_config.algorithm).strip()
    _validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    compiled_model = _maybe_compile_learner_model(
        model=model,
        training_config=training_config,
        device=device,
    )
    learner = _build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
    )
    resume_state = None
    init_schedule_offset_updates = 0
    learner.init_schedule_offset_updates = 0
    if resume_checkpoint_path is not None:
        resume_state = _restore_learner_from_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
        )
        print(
            "Resumed learner state: "
            f"checkpoint={resume_state.checkpoint_path} "
            f"update={resume_state.update_count} "
            f"policy_version={resume_state.policy_version}"
        )
        init_schedule_offset_updates = max(0, int(getattr(resume_state, "init_schedule_offset_updates", 0)))
        if init_schedule_offset_updates == 0:
            init_schedule_offset_updates = _infer_init_schedule_offset_from_scalars(training_paths.scalars_path)
            if init_schedule_offset_updates > 0:
                print(
                    "Recovered warm-start schedule offset from scalar logs: "
                    f"init_schedule_offset_updates={init_schedule_offset_updates}"
                )
        learner.init_schedule_offset_updates = init_schedule_offset_updates
    if init_from_checkpoint_path is not None:
        init_state = _initialize_learner_from_checkpoint(
            checkpoint_path=init_from_checkpoint_path,
            learner=learner,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
        )
        print(
            "Initialized learner weights: "
            f"checkpoint={init_state.checkpoint_path} "
            f"source_update={init_state.update_count} "
            f"source_init_schedule_offset={init_state.init_schedule_offset_updates} "
            f"source_policy_version={init_state.policy_version}"
        )
        init_schedule_offset_updates = _effective_init_schedule_offset_from_checkpoint(
            source_update_count=int(init_state.update_count),
            source_init_schedule_offset_updates=int(init_state.init_schedule_offset_updates),
            override_updates=init_schedule_offset_override_updates,
        )
        if init_schedule_offset_override_updates is not None:
            print(
                "Overrode init-from-checkpoint guidance schedule offset: "
                f"init_schedule_offset_updates={init_schedule_offset_updates}"
            )
        learner.init_schedule_offset_updates = init_schedule_offset_updates

    config_hash256 = compute_config_hash256(stack)
    _ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        learner=learner,
        device=device,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        baseline_run_dir=b1_baseline_run_dir,
    )
    if seed_snapshot_run_dir is not None:
        _import_seed_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=contract.spec_bundle,
        run_dir=artifacts.run_dir,
        performance_log_path=training_paths.performance_log_path,
        learner_device=device,
    )
    latest_metrics: dict[str, float] = {}
    if resume_state is not None:
        latest_metrics.update(
            _publish_initial_runtime_snapshot_after_resume(
                runtime=runtime,
                model=model,
                update_count=int(learner.update_count),
            )
        )
        print(f"Published resumed actor snapshot: update={int(learner.update_count)}")
    actor_torch_threads = _central_runtime_actor_torch_threads(stack, runtime)
    learner_torch_threads = None if stack.config.system is None else int(stack.config.system.learner_torch_threads)
    last_checkpoint_guard_rollback_update: int | None = None
    last_dev_eval_summary: Mapping[str, Any] | None = None
    last_dev_eval_update_count: int | None = None
    trajectory_bc_replay_state = TrajectoryBcReplayState.from_training_config(
        training_config,
        repo_root=stack.root,
    )
    paired_swing_replay_state = PairedSwingReplayState.from_training_config(
        training_config,
        repo_root=stack.root,
    )
    paired_outcome_preference_replay_state = PairedOutcomePreferenceReplayState.from_training_config(
        training_config,
        repo_root=stack.root,
    )
    if paired_outcome_preference_replay_state is not None and resume_state is None:
        _reset_policy_anchor_to_current_model(learner)
    start_time = time.time()
    profiler, profiler_context, profiler_trace_dir = build_training_profiler(
        enabled=bool(torch_profiler),
        run_dir=artifacts.run_dir,
        device=device,
    )
    with profiler_context:
        if int(learner.update_count) == 0:
            latest_metrics = _run_structured_warmstart(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=bool(profile_timers),
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            )
        if int(learner.update_count) >= max_updates:
            raise RuntimeError(
                f"Resume checkpoint is already at update {learner.update_count}, which is >= --max-updates {max_updates}"
            )
        try:
            for _update_index in range(int(learner.update_count), max_updates):
                schedule_update_count = _schedule_update_count_for_next_update(
                    learner_update_count=int(learner.update_count),
                    init_schedule_offset_updates=init_schedule_offset_updates,
                )
                guidance_schedule_metrics = _apply_guidance_schedule_for_next_update(
                    learner=learner,
                    model=model,
                    stack=stack,
                    update_count=schedule_update_count,
                )
                guidance_schedule_metrics["guidance_schedule_update_count"] = float(schedule_update_count)
                if init_schedule_offset_updates > 0:
                    guidance_schedule_metrics["init_schedule_offset_updates"] = float(init_schedule_offset_updates)
                learner.set_entropy_coef(
                    _entropy_coef_for_next_update(training_config, update_count=schedule_update_count)
                )
                with (
                    profile_block(profile_timers, "collect_update_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with profile_block(profile_timers, "learner_update"), _torch_num_threads_scope(learner_torch_threads):
                    latest_metrics = learner.update(runtime_batch.learner_batch)
                with (
                    profile_block(profile_timers, "trajectory_bc_replay"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    maybe_run_trajectory_bc_replay(
                        state=trajectory_bc_replay_state,
                        learner=learner,
                        training_config=training_config,
                        device=device,
                        update_count=int(learner.update_count),
                        latest_metrics=latest_metrics,
                    )
                with (
                    profile_block(profile_timers, "paired_swing_replay"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    maybe_run_paired_swing_replay(
                        state=paired_swing_replay_state,
                        learner=learner,
                        device=device,
                        update_count=int(learner.update_count),
                        latest_metrics=latest_metrics,
                    )
                with (
                    profile_block(profile_timers, "paired_outcome_preference_replay"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    maybe_run_paired_outcome_preference_replay(
                        state=paired_outcome_preference_replay_state,
                        learner=learner,
                        device=device,
                        update_count=int(learner.update_count),
                        latest_metrics=latest_metrics,
                    )
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(guidance_schedule_metrics)
                with profile_block(profile_timers, "runtime_snapshot_publish"):
                    latest_metrics.update(
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                        )
                    )
                _merge_post_update_auxiliary_metrics_into_training_log(learner=learner, metrics=latest_metrics)
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
                if learner.update_count % checkpoint_interval_updates == 0:
                    ckpt_path = training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt"
                    _write_checkpoint(
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=last_dev_eval_summary,
                    )
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))

                    if learner.model is None:
                        raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
                    candidate_policy_id = _persist_snapshot_registry_entry(
                        stack=stack,
                        training_paths=training_paths,
                        run_dir=artifacts.run_dir,
                        checkpoint_path=ckpt_path,
                        model_state_dict=learner.model.state_dict(),
                        config_hash256=config_hash256,
                        device=device,
                        update=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        model=learner.model,
                    )
                    if _is_noleague_baseline_role(_experiment_role(stack)):
                        _ensure_noleague_baseline_anchor(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            learner=learner,
                            device=device,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            permit_current_run_alias=True,
                            source_checkpoint_path=ckpt_path,
                            update=int(learner.update_count),
                        )
                    runtime.refresh_opponent_pool()
                    promotion_passed = _run_snapshot_promotion_gate(
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        learner=learner,
                        candidate_policy_id=candidate_policy_id,
                        update_count=int(learner.update_count),
                        league_reference_update=(
                            None
                            if "league_effective_update" not in latest_metrics
                            else int(latest_metrics["league_effective_update"])
                        ),
                        policy_version=int(learner.get_policy_version()),
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    if promotion_passed:
                        runtime.refresh_opponent_pool()

                if _should_run_periodic_dev_eval(stack, update_count=int(learner.update_count)):
                    summary_payload = _run_periodic_dev_eval(
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        learner=learner,
                        device=device,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    anchor_keys = sorted(cast(dict[str, Any], summary_payload["anchor_scores"]).keys())
                    opponent_fragment = f" opponent={_slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
                    print(
                        "Periodic dev eval: "
                        f"update={learner.update_count}{opponent_fragment} "
                        f"aggregate={summary_payload['aggregate_score']:.4f} "
                        f"anchors={','.join(anchor_keys)}"
                    )
                    effective_summary = summary_payload
                    tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
                    existing_best_record = tracker_before_dev_eval.get("best")
                    if not isinstance(existing_best_record, Mapping):
                        existing_best_record = None
                    confirmatory_request = _confirmatory_dev_eval_request(
                        stack=stack,
                        existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
                        dev_eval_summary=summary_payload,
                    )
                    if confirmatory_request is not None:
                        seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = (
                            _periodic_dev_eval_schedule(stack)
                        )
                        confirmatory_pairs = _expand_periodic_dev_eval_paired_seeds(
                            base_paired_seeds,
                            requested_pairs=int(confirmatory_request["target_pairs"]),
                            seed_file_sha256=seed_file_sha256,
                            update_count=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            scope="periodic_dev_eval_confirmatory",
                        )
                        effective_summary = _run_periodic_dev_eval(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            device=device,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            artifact_dir_name="dev_eval_confirmatory",
                            artifact_scope="periodic_dev_eval_confirmatory",
                            paired_seeds_override=confirmatory_pairs,
                            persist_summary=False,
                            update_stall_monitor=False,
                        )
                        print(
                            "Confirmatory dev eval: "
                            f"update={learner.update_count} paired_seeds={len(confirmatory_pairs)} "
                            f"aggregate={effective_summary['aggregate_score']:.4f} "
                            f"reasons={','.join(cast(list[str], confirmatory_request['reasons']))} "
                            f"seed_file={seed_file.name}"
                        )
                    last_dev_eval_summary = effective_summary
                    last_dev_eval_update_count = int(learner.update_count)
                    ckpt_path = _ensure_current_checkpoint(
                        training_paths=training_paths,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                    )
                    guard_event = _maybe_rollback_to_best_checkpoint(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        runtime=runtime,
                        learner=learner,
                        model=model,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                        last_rollback_update=last_checkpoint_guard_rollback_update,
                    )
                    checkpoint_guard_stop_requested = False
                    if guard_event is not None:
                        last_checkpoint_guard_rollback_update = int(learner.update_count)
                        print(
                            "Checkpoint guard rollback: "
                            f"update={guard_event['update_count']} "
                            f"best_update={guard_event['best_update_count']} "
                            f"current_score={float(guard_event['current_score']):.4f} "
                            f"best_score={float(guard_event['best_score']):.4f} "
                            f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                        )
                        curriculum = stack.config.curriculum
                        checkpoint_guard_stop_requested = bool(
                            curriculum is not None
                            and getattr(curriculum.checkpoint_guard, "stop_after_rollback", False)
                        )
                        if checkpoint_guard_stop_requested:
                            latest_metrics["checkpoint_guard_stop_after_rollback"] = 1.0
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(learner.update_count))
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
                    if checkpoint_guard_stop_requested:
                        print(
                            "Checkpoint guard early stop after rollback: "
                            f"update={guard_event['update_count']} "
                            f"best_update={guard_event['best_update_count']}"
                        )
                        break
        finally:
            runtime.close()

    if profiler is not None and profiler_trace_dir is not None:
        trace_path = profiler_trace_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(f"Wrote torch profiler trace: {trace_path}")

    if not latest_metrics:
        raise RuntimeError("The canonical single-node run finished without producing learner metrics")
    final_checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    final_dev_eval_summary = last_dev_eval_summary if last_dev_eval_update_count == int(learner.update_count) else None
    tracker_payload = _publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=final_checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    finalize_guard_event = _maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    if finalize_guard_event is not None:
        print(
            "Checkpoint guard final selection: "
            f"update={finalize_guard_event['update_count']} "
            f"best_update={finalize_guard_event['best_update_count']} "
            f"current_score={float(finalize_guard_event['current_score']):.4f} "
            f"best_score={float(finalize_guard_event['best_score']):.4f}"
        )
        tracker_payload = _load_checkpoint_tracker(training_paths)
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
    return latest_metrics
