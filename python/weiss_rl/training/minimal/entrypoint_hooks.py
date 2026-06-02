"""Compatibility hook assembly for the path-based training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.runtime.components.topology import QueueRuntimeMode


@dataclass(frozen=True)
class MinimalTrainingEntryRequest:
    stack: Any
    contract: Any
    artifacts: Any
    num_envs: int
    unroll_length: int
    max_updates: int
    profile: str
    device: torch.device
    seed: int
    checkpoint_interval_updates: int
    run_id256: str
    config_hash256: str
    spec_hash256: str
    runtime_mode: QueueRuntimeMode
    b1_baseline_run_dir: Path | None
    seed_snapshot_run_dir: Path | None = None
    profile_timers: bool = False
    torch_profiler: bool = False
    resume_checkpoint_path: Path | None = None
    init_from_checkpoint_path: Path | None = None
    init_schedule_offset_override_updates: int | None = None
    tensorboard_logger: TensorBoardLogger | None = None


def run_minimal_training_with_script_hooks(api: Any, request: MinimalTrainingEntryRequest) -> dict[str, float]:
    """Run minimal training through script-level hooks.

    The training entrypoint module intentionally exposes many
    private names that tests monkeypatch. Keeping hook assembly here but
    resolving every dependency from `api` preserves that compatibility surface.
    """

    return api.run_minimal_training(
        stack=request.stack,
        contract=request.contract,
        artifacts=request.artifacts,
        num_envs=request.num_envs,
        unroll_length=request.unroll_length,
        max_updates=request.max_updates,
        profile=request.profile,
        device=request.device,
        seed=request.seed,
        checkpoint_interval_updates=request.checkpoint_interval_updates,
        run_id256=request.run_id256,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        runtime_mode=request.runtime_mode,
        b1_baseline_run_dir=request.b1_baseline_run_dir,
        seed_snapshot_run_dir=request.seed_snapshot_run_dir,
        profile_timers=request.profile_timers,
        torch_profiler=request.torch_profiler,
        resume_checkpoint_path=request.resume_checkpoint_path,
        init_from_checkpoint_path=request.init_from_checkpoint_path,
        init_schedule_offset_override_updates=request.init_schedule_offset_override_updates,
        tensorboard_logger=request.tensorboard_logger,
        hooks=api.MinimalTrainingHooks(
            configure_torch_threads=api._configure_torch_threads,
            spec_dimensions=api._spec_dimensions,
            experiment_role=api._experiment_role,
            training_paths=api._training_paths,
            validate_algorithm_model_contract=api._validate_algorithm_model_contract,
            build_policy_value_model=api.build_policy_value_model,
            maybe_compile_learner_model=api._maybe_compile_learner_model,
            build_training_learner=api._build_training_learner,
            restore_learner_from_checkpoint=api._restore_learner_from_checkpoint,
            initialize_learner_from_checkpoint=api._initialize_learner_from_checkpoint,
            compute_config_hash256=api.compute_config_hash256,
            ensure_noleague_baseline_anchor=api._ensure_noleague_baseline_anchor,
            import_seed_snapshot_pool=api._import_seed_snapshot_pool,
            canonical_config_dict=api.canonical_config_dict,
            build_runtime_config=api.build_runtime_config,
            queue_runtime_cls=api.QueueRuntime,
            central_runtime_actor_torch_threads=api._central_runtime_actor_torch_threads,
            build_training_profiler=api.build_training_profiler,
            run_structured_warmstart=api._run_structured_warmstart,
            profile_block=api.profile_block,
            apply_guidance_schedule_for_next_update=api._apply_guidance_schedule_for_next_update,
            entropy_coef_for_next_update=api._entropy_coef_for_next_update,
            torch_num_threads_scope=api._torch_num_threads_scope,
            collect_training_batch=api.collect_training_batch,
            write_scalars_record=api._write_scalars_record,
            write_checkpoint=api._write_checkpoint,
            publish_checkpoint_aliases=api._publish_checkpoint_aliases,
            maybe_log_structured_mainmove_guard=api._maybe_log_structured_mainmove_guard,
            persist_snapshot_registry_entry=api._persist_snapshot_registry_entry,
            is_noleague_baseline_role=api._is_noleague_baseline_role,
            run_snapshot_promotion_gate=api._run_snapshot_promotion_gate,
            should_run_periodic_dev_eval=api._should_run_periodic_dev_eval,
            run_periodic_dev_eval=api._run_periodic_dev_eval,
            slug_policy_id=api._slug_policy_id,
            load_checkpoint_tracker=api._load_checkpoint_tracker,
            confirmatory_dev_eval_request=api._confirmatory_dev_eval_request,
            periodic_dev_eval_schedule=api._periodic_dev_eval_schedule,
            expand_periodic_dev_eval_paired_seeds=api._expand_periodic_dev_eval_paired_seeds,
            ensure_current_checkpoint=api._ensure_current_checkpoint,
            maybe_rollback_to_best_checkpoint=api._maybe_rollback_to_best_checkpoint,
            maybe_finalize_from_best_checkpoint=api._maybe_finalize_from_best_checkpoint,
        ),
    )
