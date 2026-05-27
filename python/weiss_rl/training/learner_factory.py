from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import nn

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner

from .batches import IMPALA_ALGORITHMS, PPO_ALGORITHMS


def _common_training_learner_kwargs(
    *,
    model: Any,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: Any,
    pass_action_id: int,
    checkpoint_interval_updates: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "compiled_model": compiled_model,
        "learning_rate": training_config.learning_rate,
        "value_loss_coef": training_config.value_loss_coef,
        "entropy_coef": training_config.entropy_coef,
        "grad_norm_clip": training_config.grad_norm_clip,
        "mixed_precision": bool(training_config.mixed_precision),
        "checkpoint_dir": training_paths.checkpoints_dir,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
        "teacher_family_coef": training_config.teacher_family_coef,
        "teacher_slot_coef": training_config.teacher_slot_coef,
        "teacher_hand_coef": training_config.teacher_hand_coef,
        "teacher_move_source_coef": training_config.teacher_move_source_coef,
        "teacher_attack_type_coef": training_config.teacher_attack_type_coef,
        "teacher_action_coef": training_config.teacher_action_coef,
        "teacher_same_family_action_coef": training_config.teacher_same_family_action_coef,
        "teacher_action_margin_coef": training_config.teacher_action_margin_coef,
        "teacher_action_margin": training_config.teacher_action_margin,
        "teacher_same_family_action_margin_coef": training_config.teacher_same_family_action_margin_coef,
        "teacher_same_family_action_margin": training_config.teacher_same_family_action_margin,
        "teacher_exact_action_families": training_config.teacher_exact_action_families,
        "teacher_public_heuristic_coef": training_config.teacher_public_heuristic_coef,
        "teacher_public_heuristic_temperature": training_config.teacher_public_heuristic_temperature,
        "teacher_public_nonpass_over_pass_coef": training_config.teacher_public_nonpass_over_pass_coef,
        "teacher_public_nonpass_over_pass_margin": training_config.teacher_public_nonpass_over_pass_margin,
        "teacher_public_heuristic_families": training_config.teacher_public_heuristic_families,
        "teacher_public_heuristic_profiles": training_config.teacher_public_heuristic_profiles,
        "teacher_public_heuristic_profile_mode": training_config.teacher_public_heuristic_profile_mode,
        "teacher_public_heuristic_profiles_end_updates": training_config.teacher_public_heuristic_profiles_end_updates,
        "policy_anchor_coef": training_config.policy_anchor_coef,
        "policy_anchor_top_action_coef": training_config.policy_anchor_top_action_coef,
        "policy_anchor_temperature": training_config.policy_anchor_temperature,
        "trajectory_retention_coef": training_config.trajectory_retention_coef,
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
    }


def build_training_learner(
    *,
    algorithm: str,
    model: Any,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: Any,
    pass_action_id: int,
    checkpoint_interval_updates: int,
    impala_learner_cls: Callable[..., Any] = ImpalaLearner,
    ppo_lite_learner_cls: Callable[..., Any] = PpoLiteLearner,
) -> ImpalaLearner | PpoLiteLearner:
    """Construct the training learner for the configured algorithm family."""

    common_kwargs = _common_training_learner_kwargs(
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
    )
    if algorithm in IMPALA_ALGORITHMS:
        return cast(
            ImpalaLearner,
            impala_learner_cls(
                **common_kwargs,
                entropy_scope=str(getattr(training_config, "entropy_scope", "candidate")),
                vtrace_rho_bar=training_config.vtrace_rho_bar,
                vtrace_c_bar=training_config.vtrace_c_bar,
            ),
        )
    if algorithm in PPO_ALGORITHMS:
        if float(getattr(training_config, "trajectory_retention_coef", 0.0)) != 0.0:
            raise RuntimeError("training.structured_aux.trajectory_retention_coef is only supported by IMPALA/V-trace")
        return cast(
            PpoLiteLearner,
            ppo_lite_learner_cls(
                **common_kwargs,
                ppo_clip_epsilon=training_config.ppo_clip_epsilon,
                value_clip_epsilon=training_config.ppo_value_clip_epsilon,
                ppo_epochs=int(training_config.ppo_epochs),
                target_kl=training_config.ppo_target_kl,
                normalize_advantages=bool(training_config.ppo_normalize_advantages),
            ),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")
