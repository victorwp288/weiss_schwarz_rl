"""Context construction for canonical minimal training runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.loop.loop_progress import TrainingLoopProgress
from weiss_rl.training.loop.post_update import (
    FinalTrainingCheckpointContext,
    FinalTrainingCheckpointHooks,
    PostUpdateCheckpointDevEvalContext,
    PostUpdateCheckpointDevEvalHooks,
    PostUpdateCheckpointDevEvalSchedule,
)
from weiss_rl.training.loop.setup import MinimalTrainingSetup
from weiss_rl.training.loop.update_step import (
    TrainingUpdateStepHooks,
    TrainingUpdateStepInputs,
    TrainingUpdateStepOptions,
)


@dataclass(frozen=True, slots=True)
class TrainingRunContexts:
    progress: TrainingLoopProgress
    update_inputs: TrainingUpdateStepInputs
    update_options: TrainingUpdateStepOptions
    update_hooks: TrainingUpdateStepHooks
    post_update_context: PostUpdateCheckpointDevEvalContext
    post_update_schedule: PostUpdateCheckpointDevEvalSchedule
    post_update_hooks: PostUpdateCheckpointDevEvalHooks
    final_checkpoint_context: FinalTrainingCheckpointContext
    final_checkpoint_hooks: FinalTrainingCheckpointHooks
    actor_torch_threads: int | None
    learner_torch_threads: int | None


def build_training_run_contexts(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    setup: MinimalTrainingSetup,
    profile_timers: bool,
    device: torch.device,
    checkpoint_interval_updates: int,
    run_id256: str,
    spec_hash256: str,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: Any,
    replay_states_from_config: Callable[..., Any],
    reset_policy_anchor: Callable[..., None],
    checkpoint_fn: Callable[..., Any],
    dev_eval_fn: Callable[..., Any],
    finalize_fn: Callable[..., Any],
) -> TrainingRunContexts:
    learner = setup.learner
    model = setup.model
    runtime = setup.runtime
    training_config = setup.training_config
    rewards_config = setup.rewards_config
    training_paths = setup.training_paths
    algorithm = setup.algorithm
    actor_torch_threads = hooks.central_runtime_actor_torch_threads(stack, runtime)
    learner_torch_threads = None if stack.config.system is None else int(stack.config.system.learner_torch_threads)
    replay_states = replay_states_from_config(
        training_config,
        repo_root=stack.root,
    )
    reset_policy_anchor(
        learner=learner,
        replay_states=replay_states,
        resume_state=setup.resume_state,
    )

    return TrainingRunContexts(
        progress=TrainingLoopProgress(latest_metrics=setup.latest_metrics),
        update_inputs=TrainingUpdateStepInputs(
            learner=learner,
            model=model,
            stack=stack,
            runtime=runtime,
            algorithm=algorithm,
            training_config=training_config,
            rewards_config=rewards_config,
            replay_states=replay_states,
            device=device,
            init_schedule_offset_updates=setup.init_schedule_offset_updates,
        ),
        update_options=TrainingUpdateStepOptions(
            profile_timers=bool(profile_timers),
            actor_torch_threads=actor_torch_threads,
            learner_torch_threads=learner_torch_threads,
        ),
        update_hooks=TrainingUpdateStepHooks(
            apply_guidance_schedule_for_next_update=hooks.apply_guidance_schedule_for_next_update,
            entropy_coef_for_next_update=hooks.entropy_coef_for_next_update,
            collect_training_batch=hooks.collect_training_batch,
            profile_block=hooks.profile_block,
            torch_num_threads_scope=hooks.torch_num_threads_scope,
        ),
        post_update_context=PostUpdateCheckpointDevEvalContext(
            learner=learner,
            model=model,
            stack=stack,
            contract=contract,
            artifacts=artifacts,
            training_paths=training_paths,
            runtime=runtime,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
            run_id256=run_id256,
            config_hash256=setup.config_hash256,
            tensorboard_logger=tensorboard_logger,
        ),
        post_update_schedule=PostUpdateCheckpointDevEvalSchedule(
            checkpoint_interval_updates=checkpoint_interval_updates,
        ),
        post_update_hooks=PostUpdateCheckpointDevEvalHooks(
            checkpoint_hooks=hooks.checkpoint_promotion,
            periodic_dev_eval_hooks=hooks.periodic_dev_eval,
            checkpoint_fn=checkpoint_fn,
            dev_eval_fn=dev_eval_fn,
        ),
        final_checkpoint_context=FinalTrainingCheckpointContext(
            learner=learner,
            stack=stack,
            artifacts=artifacts,
            training_paths=training_paths,
            runtime=runtime,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
            tensorboard_logger=tensorboard_logger,
        ),
        final_checkpoint_hooks=FinalTrainingCheckpointHooks(
            hooks=hooks.final_checkpoint,
            finalize_fn=finalize_fn,
        ),
        actor_torch_threads=actor_torch_threads,
        learner_torch_threads=learner_torch_threads,
    )


__all__ = [
    "TrainingRunContexts",
    "build_training_run_contexts",
]
