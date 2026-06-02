"""Execution phase for the canonical minimal training loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.finalization import (
    TrainingFinalCheckpointHooks,
    finalize_training_checkpoint_selection,
)
from weiss_rl.training.checkpointing.periodic_dev_eval import (
    TrainingPeriodicDevEvalHooks,
    maybe_run_periodic_dev_eval_and_checkpoint_guard,
)
from weiss_rl.training.checkpointing.snapshot_promotion import (
    TrainingCheckpointPromotionHooks,
    maybe_checkpoint_and_promote_snapshot,
)
from weiss_rl.training.loop.loop_progress import (
    write_training_update_outputs,
)
from weiss_rl.training.loop.post_update import (
    finalize_training_loop_progress_from_context,
    run_post_update_checkpoint_and_dev_eval_from_context,
)
from weiss_rl.training.loop.run_contexts import build_training_run_contexts
from weiss_rl.training.loop.setup import MinimalTrainingSetup
from weiss_rl.training.loop.update_step import run_training_update_step_from_context
from weiss_rl.training.replay_data.training_replay_dispatch import (
    reset_policy_anchor_for_fresh_preference_replay,
    training_replay_states_from_config,
)


@dataclass(frozen=True, slots=True)
class MinimalTrainingRunHooks:
    central_runtime_actor_torch_threads: Any
    build_training_profiler: Any
    run_structured_warmstart: Any
    profile_block: Any
    apply_guidance_schedule_for_next_update: Any
    entropy_coef_for_next_update: Any
    torch_num_threads_scope: Any
    collect_training_batch: Any
    write_scalars_record: Any
    checkpoint_promotion: TrainingCheckpointPromotionHooks
    periodic_dev_eval: TrainingPeriodicDevEvalHooks
    final_checkpoint: TrainingFinalCheckpointHooks


def run_minimal_training_updates(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    setup: MinimalTrainingSetup,
    max_updates: int,
    profile_timers: bool,
    torch_profiler: bool,
    device: torch.device,
    checkpoint_interval_updates: int,
    run_id256: str,
    spec_hash256: str,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: MinimalTrainingRunHooks,
) -> dict[str, float]:
    learner = setup.learner
    runtime = setup.runtime
    training_config = setup.training_config
    rewards_config = setup.rewards_config
    training_paths = setup.training_paths
    algorithm = setup.algorithm
    run_contexts = build_training_run_contexts(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        setup=setup,
        profile_timers=profile_timers,
        device=device,
        checkpoint_interval_updates=checkpoint_interval_updates,
        run_id256=run_id256,
        spec_hash256=spec_hash256,
        tensorboard_logger=tensorboard_logger,
        hooks=hooks,
        replay_states_from_config=training_replay_states_from_config,
        reset_policy_anchor=reset_policy_anchor_for_fresh_preference_replay,
        checkpoint_fn=maybe_checkpoint_and_promote_snapshot,
        dev_eval_fn=maybe_run_periodic_dev_eval_and_checkpoint_guard,
        finalize_fn=finalize_training_checkpoint_selection,
    )
    progress = run_contexts.progress
    start_time = time.time()
    profiler, profiler_context, profiler_trace_dir = hooks.build_training_profiler(
        enabled=bool(torch_profiler),
        run_dir=artifacts.run_dir,
        device=device,
    )
    with profiler_context:
        if int(learner.update_count) == 0:
            progress.record_latest_metrics(
                hooks.run_structured_warmstart(
                    learner=learner,
                    runtime=runtime,
                    algorithm=algorithm,
                    training_config=training_config,
                    rewards_config=rewards_config,
                    training_paths=training_paths,
                    tensorboard_logger=tensorboard_logger,
                    start_time=start_time,
                    profile_timers=bool(profile_timers),
                    actor_torch_threads=run_contexts.actor_torch_threads,
                    learner_torch_threads=run_contexts.learner_torch_threads,
                )
            )
        if int(learner.update_count) >= max_updates:
            raise RuntimeError(
                f"Resume checkpoint is already at update {learner.update_count}, which is >= --max-updates "
                f"{max_updates}"
            )
        try:
            for _update_index in range(int(learner.update_count), max_updates):
                progress.record_latest_metrics(
                    run_training_update_step_from_context(
                        inputs=run_contexts.update_inputs,
                        options=run_contexts.update_options,
                        hooks=run_contexts.update_hooks,
                    )
                )
                write_training_update_outputs(
                    progress=progress,
                    learner=learner,
                    training_paths=training_paths,
                    start_time=start_time,
                    tensorboard_logger=tensorboard_logger,
                    write_scalars_record=hooks.write_scalars_record,
                )
                stop_requested = run_post_update_checkpoint_and_dev_eval_from_context(
                    progress=progress,
                    context=run_contexts.post_update_context,
                    schedule=run_contexts.post_update_schedule,
                    hooks=run_contexts.post_update_hooks,
                )
                if stop_requested:
                    break
        finally:
            runtime.close()

    if profiler is not None and profiler_trace_dir is not None:
        trace_path = profiler_trace_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(f"Wrote torch profiler trace: {trace_path}")

    if not progress.latest_metrics:
        raise RuntimeError("The canonical single-node run finished without producing learner metrics")
    finalize_training_loop_progress_from_context(
        progress=progress,
        context=run_contexts.final_checkpoint_context,
        hooks=run_contexts.final_checkpoint_hooks,
    )
    return progress.latest_metrics


__all__ = [
    "MinimalTrainingRunHooks",
    "run_minimal_training_updates",
]
