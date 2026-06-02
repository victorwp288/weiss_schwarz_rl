"""Setup phase for the canonical minimal training loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.runtime import QueueRuntimeMode


@dataclass(frozen=True, slots=True)
class MinimalTrainingSetupHooks:
    spec_dimensions: Any
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


@dataclass(frozen=True, slots=True)
class MinimalTrainingSetup:
    observation_dim: int
    action_dim: int
    training_config: Any
    rewards_config: Any
    training_paths: Any
    pass_action_id: int
    algorithm: str
    model: Any
    learner: Any
    runtime: Any
    latest_metrics: dict[str, float]
    init_schedule_offset_updates: int
    resume_state: Any | None
    config_hash256: str


def require_training_stack_components(stack: StackConfig) -> tuple[Any, Any, Any, Any]:
    training_config = stack.config.training
    model_config = stack.config.model
    environment_config = stack.config.environment
    rewards_config = stack.config.rewards
    if training_config is None or model_config is None or environment_config is None or rewards_config is None:
        raise RuntimeError("The locked stack is missing training, model, environment, or rewards config")
    return training_config, model_config, environment_config, rewards_config


def build_minimal_training_setup(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    profile: str,
    device: torch.device,
    seed: int,
    checkpoint_interval_updates: int,
    spec_hash256: str,
    runtime_mode: QueueRuntimeMode,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    resume_checkpoint_path: Path | None,
    init_from_checkpoint_path: Path | None,
    init_schedule_offset_override_updates: int | None,
    hooks: MinimalTrainingSetupHooks,
) -> MinimalTrainingSetup:
    observation_dim, action_dim = hooks.spec_dimensions(contract)
    training_config, model_config, _environment_config, rewards_config = require_training_stack_components(stack)

    training_paths = hooks.training_paths(artifacts.run_dir)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    algorithm = str(training_config.algorithm).strip()
    hooks.validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    model = hooks.build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    compiled_model = hooks.maybe_compile_learner_model(
        model=model,
        training_config=training_config,
        device=device,
    )
    learner = hooks.build_training_learner(
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
        resume_state = hooks.restore_learner_from_checkpoint(
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
            init_schedule_offset_updates = infer_init_schedule_offset_from_scalars(training_paths.scalars_path)
            if init_schedule_offset_updates > 0:
                print(
                    "Recovered warm-start schedule offset from scalar logs: "
                    f"init_schedule_offset_updates={init_schedule_offset_updates}"
                )
        learner.init_schedule_offset_updates = init_schedule_offset_updates
    if init_from_checkpoint_path is not None:
        init_state = hooks.initialize_learner_from_checkpoint(
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
        init_schedule_offset_updates = effective_init_schedule_offset_from_checkpoint(
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

    config_hash256 = hooks.compute_config_hash256(stack)
    hooks.ensure_noleague_baseline_anchor(
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
        hooks.import_seed_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=hooks.canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
    runtime_config = hooks.build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
    )
    runtime = hooks.queue_runtime_cls(
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
            publish_initial_runtime_snapshot_after_resume(
                runtime=runtime,
                model=model,
                update_count=int(learner.update_count),
            )
        )
        print(f"Published resumed actor snapshot: update={int(learner.update_count)}")

    return MinimalTrainingSetup(
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        training_config=training_config,
        rewards_config=rewards_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        algorithm=algorithm,
        model=model,
        learner=learner,
        runtime=runtime,
        latest_metrics=latest_metrics,
        init_schedule_offset_updates=init_schedule_offset_updates,
        resume_state=resume_state,
        config_hash256=config_hash256,
    )


def publish_initial_runtime_snapshot_after_resume(*, runtime: Any, model: Any, update_count: int) -> dict[str, float]:
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


def effective_init_schedule_offset_from_checkpoint(
    *,
    source_update_count: int,
    source_init_schedule_offset_updates: int,
    override_updates: int | None = None,
) -> int:
    """Carry cumulative schedule time across fresh warm-start segments."""

    if override_updates is not None:
        return max(0, int(override_updates))
    return max(0, int(source_init_schedule_offset_updates)) + max(0, int(source_update_count))


def infer_init_schedule_offset_from_scalars(scalars_path: Path) -> int:
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


__all__ = [
    "MinimalTrainingSetup",
    "MinimalTrainingSetupHooks",
    "build_minimal_training_setup",
    "effective_init_schedule_offset_from_checkpoint",
    "infer_init_schedule_offset_from_scalars",
    "publish_initial_runtime_snapshot_after_resume",
    "require_training_stack_components",
]
