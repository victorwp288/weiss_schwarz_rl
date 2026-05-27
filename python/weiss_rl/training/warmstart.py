from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .batches import IMPALA_ALGORITHMS, collect_training_batch
from .checkpoints import write_scalars_record
from .profiling import profile_block
from .torch_threads import torch_num_threads_scope


def run_structured_warmstart(
    *,
    learner: Any,
    runtime: Any,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    training_paths: Any,
    tensorboard_logger: Any | None,
    start_time: float,
    profile_timers: bool = False,
    actor_torch_threads: int | None = None,
    learner_torch_threads: int | None = None,
    collect_training_batch_fn: Callable[..., Any] = collect_training_batch,
    write_scalars_record_fn: Callable[..., Any] = write_scalars_record,
    time_fn: Callable[[], float] = time.time,
) -> dict[str, float]:
    if not bool(getattr(training_config, "structured_warmstart_enabled", False)):
        return {}
    if algorithm not in IMPALA_ALGORITHMS:
        raise RuntimeError("structured warmstart currently supports only IMPALA learners")
    warmstart_cfg = training_config.structured_warmstart
    updates = int(warmstart_cfg.updates)
    if updates <= 0:
        return {}

    previous_family = float(training_config.teacher_family_coef)
    previous_slot = float(training_config.teacher_slot_coef)
    previous_hand = float(getattr(training_config, "teacher_hand_coef", 0.0))
    previous_move_source = float(training_config.teacher_move_source_coef)
    previous_attack_type = float(training_config.teacher_attack_type_coef)
    previous_action = float(training_config.teacher_action_coef)
    previous_same_family_action = float(training_config.teacher_same_family_action_coef)
    previous_public_heuristic = float(training_config.teacher_public_heuristic_coef)
    previous_public_heuristic_temperature = float(training_config.teacher_public_heuristic_temperature)
    previous_public_heuristic_families = tuple(training_config.teacher_public_heuristic_families)
    previous_public_heuristic_profiles = tuple(training_config.teacher_public_heuristic_profiles)
    previous_public_heuristic_profile_mode = str(training_config.teacher_public_heuristic_profile_mode)
    previous_public_heuristic_profiles_end_updates = int(training_config.teacher_public_heuristic_profiles_end_updates)
    learner.set_teacher_aux_coefs(
        family=float(warmstart_cfg.teacher_family_coef),
        slot=float(warmstart_cfg.teacher_slot_coef),
        hand=float(getattr(warmstart_cfg, "teacher_hand_coef", 0.0)),
        move_source=float(warmstart_cfg.teacher_move_source_coef),
        attack_type=float(warmstart_cfg.teacher_attack_type_coef),
        action=float(warmstart_cfg.teacher_action_coef),
        same_family_action=float(warmstart_cfg.teacher_same_family_action_coef),
        public_heuristic=float(warmstart_cfg.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(warmstart_cfg.teacher_public_heuristic_temperature),
        public_heuristic_families=tuple(warmstart_cfg.teacher_public_heuristic_families),
        public_heuristic_profiles=tuple(warmstart_cfg.teacher_public_heuristic_profiles),
        public_heuristic_profile_mode=str(warmstart_cfg.teacher_public_heuristic_profile_mode),
        public_heuristic_profiles_end_updates=int(warmstart_cfg.teacher_public_heuristic_profiles_end_updates),
    )
    latest_metrics: dict[str, float] = {}
    try:
        with (
            runtime.structured_warmstart_source_mix() as warmstart_source_metrics,
            runtime.disable_mirror_policy_fusion(),
        ):
            for warmstart_step in range(updates):
                with (
                    profile_block(profile_timers, "collect_training_batch"),
                    torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = collect_training_batch_fn(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with (
                    profile_block(profile_timers, "learner_auxiliary_update"),
                    torch_num_threads_scope(learner_torch_threads),
                ):
                    latest_metrics = learner.auxiliary_update(runtime_batch.learner_batch)
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(warmstart_source_metrics)
                latest_metrics["warmstart_phase"] = 1.0
                latest_metrics["warmstart_step"] = float(warmstart_step + 1)
                write_scalars_record_fn(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time_fn() - start_time,
                        metrics=latest_metrics,
                    )
    finally:
        learner.set_teacher_aux_coefs(
            family=previous_family,
            slot=previous_slot,
            hand=previous_hand,
            move_source=previous_move_source,
            attack_type=previous_attack_type,
            action=previous_action,
            same_family_action=previous_same_family_action,
            public_heuristic=previous_public_heuristic,
            public_heuristic_temperature=previous_public_heuristic_temperature,
            public_heuristic_families=previous_public_heuristic_families,
            public_heuristic_profiles=previous_public_heuristic_profiles,
            public_heuristic_profile_mode=previous_public_heuristic_profile_mode,
            public_heuristic_profiles_end_updates=previous_public_heuristic_profiles_end_updates,
        )
    return latest_metrics
