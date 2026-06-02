"""Canonical minimal single-node training loop used by ``scripts.train``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.runtime import QueueRuntimeMode
from weiss_rl.training.loop.runner import run_minimal_training_updates
from weiss_rl.training.loop.setup import build_minimal_training_setup
from weiss_rl.training.minimal.hook_groups import minimal_training_hook_groups


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
    hook_groups = minimal_training_hook_groups(hooks)
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    setup = build_minimal_training_setup(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        device=device,
        seed=seed,
        checkpoint_interval_updates=checkpoint_interval_updates,
        spec_hash256=spec_hash256,
        runtime_mode=runtime_mode,
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        init_from_checkpoint_path=init_from_checkpoint_path,
        init_schedule_offset_override_updates=init_schedule_offset_override_updates,
        hooks=hook_groups.setup,
    )
    return run_minimal_training_updates(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        setup=setup,
        max_updates=max_updates,
        profile_timers=profile_timers,
        torch_profiler=torch_profiler,
        device=device,
        checkpoint_interval_updates=checkpoint_interval_updates,
        run_id256=run_id256,
        spec_hash256=spec_hash256,
        tensorboard_logger=tensorboard_logger,
        hooks=hook_groups.run,
    )
