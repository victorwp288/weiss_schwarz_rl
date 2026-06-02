from __future__ import annotations

from typing import Any

from torch import nn

_PPO_TRAJECTORY_RETENTION_ERROR = (
    "training.structured_aux.trajectory_retention_coef is only supported by IMPALA/V-trace"
)


def model_and_optimizer_kwargs(
    *,
    model: Any,
    compiled_model: nn.Module | None,
    training_config: Any,
) -> dict[str, Any]:
    return {
        "model": model,
        "compiled_model": compiled_model,
        "learning_rate": training_config.learning_rate,
        "value_loss_coef": training_config.value_loss_coef,
        "entropy_coef": training_config.entropy_coef,
        "grad_norm_clip": training_config.grad_norm_clip,
        "mixed_precision": bool(training_config.mixed_precision),
    }


def checkpoint_and_logging_kwargs(
    *,
    training_paths: Any,
    pass_action_id: int,
    checkpoint_interval_updates: int,
) -> dict[str, Any]:
    return {
        "checkpoint_dir": training_paths.checkpoints_dir,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
    }


def teacher_supervision_kwargs(training_config: Any) -> dict[str, Any]:
    return {
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
    }


def policy_anchor_kwargs(training_config: Any) -> dict[str, Any]:
    return {
        "policy_anchor_coef": training_config.policy_anchor_coef,
        "policy_anchor_top_action_coef": training_config.policy_anchor_top_action_coef,
        "policy_anchor_temperature": training_config.policy_anchor_temperature,
    }


def auxiliary_runtime_kwargs(training_config: Any) -> dict[str, Any]:
    return {
        "trajectory_retention_coef": training_config.trajectory_retention_coef,
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
    }


def common_training_learner_kwargs(
    *,
    model: Any,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: Any,
    pass_action_id: int,
    checkpoint_interval_updates: int,
) -> dict[str, Any]:
    return {
        **model_and_optimizer_kwargs(
            model=model,
            compiled_model=compiled_model,
            training_config=training_config,
        ),
        **checkpoint_and_logging_kwargs(
            training_paths=training_paths,
            pass_action_id=pass_action_id,
            checkpoint_interval_updates=checkpoint_interval_updates,
        ),
        **teacher_supervision_kwargs(training_config),
        **policy_anchor_kwargs(training_config),
        **auxiliary_runtime_kwargs(training_config),
    }


def impala_training_learner_kwargs(training_config: Any) -> dict[str, Any]:
    return {
        "entropy_scope": str(getattr(training_config, "entropy_scope", "candidate")),
        "vtrace_rho_bar": training_config.vtrace_rho_bar,
        "vtrace_c_bar": training_config.vtrace_c_bar,
    }


def ppo_training_learner_kwargs(training_config: Any) -> dict[str, Any]:
    if float(getattr(training_config, "trajectory_retention_coef", 0.0)) != 0.0:
        raise RuntimeError(_PPO_TRAJECTORY_RETENTION_ERROR)
    return {
        "ppo_clip_epsilon": training_config.ppo_clip_epsilon,
        "value_clip_epsilon": training_config.ppo_value_clip_epsilon,
        "ppo_epochs": int(training_config.ppo_epochs),
        "target_kl": training_config.ppo_target_kl,
        "normalize_advantages": bool(training_config.ppo_normalize_advantages),
    }


__all__ = [
    "auxiliary_runtime_kwargs",
    "checkpoint_and_logging_kwargs",
    "common_training_learner_kwargs",
    "impala_training_learner_kwargs",
    "model_and_optimizer_kwargs",
    "policy_anchor_kwargs",
    "ppo_training_learner_kwargs",
    "teacher_supervision_kwargs",
]
