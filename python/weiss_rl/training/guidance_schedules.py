"""Guidance coefficient schedules for training updates."""

from __future__ import annotations

from typing import Any

from weiss_rl.schedules import linear_anneal_value


def entropy_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    start = float(training_config.entropy_coef)
    target = float(training_config.entropy_anneal_to)
    steps = max(1, int(training_config.entropy_anneal_steps_updates))
    progress = min(max(int(update_count), 0), steps) / float(steps)
    return float(start + (target - start) * progress)


def teacher_public_heuristic_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(training_config.teacher_public_heuristic_coef),
            final_value=float(getattr(training_config, "teacher_public_heuristic_final_coef", 0.0)),
            start_update=int(getattr(training_config, "teacher_public_heuristic_start_updates", 0)),
            end_update=int(getattr(training_config, "teacher_public_heuristic_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def reference_policy_top_action_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    initial = float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(training_config, "reference_policy_top_action_bc_final_coef", initial)),
            start_update=int(getattr(training_config, "reference_policy_top_action_bc_start_updates", 0)),
            end_update=int(getattr(training_config, "reference_policy_top_action_bc_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def reference_policy_top_action_family_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    initial = float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(training_config, "reference_policy_top_action_family_bc_final_coef", initial)),
            start_update=int(getattr(training_config, "reference_policy_top_action_family_bc_start_updates", 0)),
            end_update=int(getattr(training_config, "reference_policy_top_action_family_bc_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def raw_b1_distill_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    if raw_b1_distill is None or not bool(getattr(raw_b1_distill, "enabled", False)):
        return 0.0
    initial = float(getattr(raw_b1_distill, "coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(raw_b1_distill, "final_coef", initial)),
            start_update=int(getattr(raw_b1_distill, "start_updates", 0)),
            end_update=int(getattr(raw_b1_distill, "end_updates", -1)),
            update_count=int(update_count),
        )
    )


def counterfactual_positive_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    counterfactual_positive = getattr(training_config, "counterfactual_positive", None)
    if counterfactual_positive is None or not bool(getattr(counterfactual_positive, "enabled", False)):
        return 0.0
    initial = float(getattr(counterfactual_positive, "coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(counterfactual_positive, "final_coef", initial)),
            start_update=int(getattr(counterfactual_positive, "start_updates", 0)),
            end_update=int(getattr(counterfactual_positive, "end_updates", -1)),
            update_count=int(update_count),
        )
    )


def public_heuristic_logit_bias_scale_for_next_update(model_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(getattr(model_config, "public_heuristic_logit_bias_scale", 0.0)),
            final_value=float(
                getattr(
                    model_config,
                    "public_heuristic_logit_bias_final_scale",
                    getattr(model_config, "public_heuristic_logit_bias_scale", 0.0),
                )
            ),
            start_update=int(getattr(model_config, "public_heuristic_logit_bias_start_updates", 0)),
            end_update=int(getattr(model_config, "public_heuristic_logit_bias_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def public_heuristic_actor_logit_bias_scale_for_next_update(
    model_config: Any,
    *,
    learner_bias_scale: float,
) -> float:
    configured_actor_scale = float(getattr(model_config, "public_heuristic_actor_logit_bias_scale", -1.0))
    if configured_actor_scale < 0.0:
        return float(learner_bias_scale)
    return configured_actor_scale


def apply_guidance_schedule_for_next_update(
    *,
    learner: Any,
    model: Any | None,
    stack: Any,
    update_count: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    training_config = stack.config.training
    if training_config is not None:
        teacher_coef = teacher_public_heuristic_coef_for_next_update(training_config, update_count=update_count)
        learner.set_teacher_aux_coefs(public_heuristic=teacher_coef)
        metrics["teacher_public_heuristic_coef_active"] = float(teacher_coef)
        reference_top_action_coef = reference_policy_top_action_bc_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        reference_family_coef = reference_policy_top_action_family_bc_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        learner.set_reference_policy_bc_coefs(
            top_action=reference_top_action_coef,
            top_action_family=reference_family_coef,
        )
        raw_b1_distill_coef = raw_b1_distill_coef_for_next_update(training_config, update_count=update_count)
        if hasattr(learner, "set_raw_b1_distill_coef"):
            learner.set_raw_b1_distill_coef(raw_b1_distill_coef)
        counterfactual_positive_coef = counterfactual_positive_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        if hasattr(learner, "set_counterfactual_positive_coef"):
            learner.set_counterfactual_positive_coef(counterfactual_positive_coef)
        metrics["reference_policy_top_action_bc_coef_active"] = float(reference_top_action_coef)
        metrics["reference_policy_top_action_family_bc_coef_active"] = float(reference_family_coef)
        metrics["raw_b1_distill_coef_active"] = float(raw_b1_distill_coef)
        metrics["counterfactual_positive_coef_active"] = float(counterfactual_positive_coef)
    model_config = stack.config.model
    if model is not None and model_config is not None:
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        if callable(set_bias_scale):
            learner_bias_scale = public_heuristic_logit_bias_scale_for_next_update(
                model_config,
                update_count=update_count,
            )
            actor_bias_scale = public_heuristic_actor_logit_bias_scale_for_next_update(
                model_config,
                learner_bias_scale=learner_bias_scale,
            )
            set_bias_scale(learner_bias_scale, actor_value=actor_bias_scale)
            metrics["public_heuristic_logit_bias_scale_active"] = float(learner_bias_scale)
            metrics["public_heuristic_actor_logit_bias_scale_active"] = float(actor_bias_scale)
    return metrics


def model_guidance_payload(model: Any | None) -> dict[str, float]:
    if model is None:
        return {}
    get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale):
        return {}
    return {
        "public_heuristic_logit_bias_scale": float(get_bias_scale(scoring_mode="learner")),
        "public_heuristic_actor_logit_bias_scale": float(get_bias_scale(scoring_mode="actor")),
    }


def restore_model_guidance_from_payload(
    model: Any | None,
    payload: dict[str, Any] | Any,
) -> None:
    if model is None:
        return
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if not callable(set_bias_scale):
        return
    learner_scale = payload.get("public_heuristic_logit_bias_scale")
    actor_scale = payload.get("public_heuristic_actor_logit_bias_scale")
    if learner_scale is None and actor_scale is None:
        return
    resolved_learner_scale = None if learner_scale is None else float(learner_scale)
    resolved_actor_scale = None if actor_scale is None else float(actor_scale)
    if resolved_learner_scale is None and resolved_actor_scale is not None:
        current_learner_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if callable(current_learner_scale):
            resolved_learner_scale = float(current_learner_scale(scoring_mode="learner"))
    if resolved_learner_scale is None:
        return
    set_bias_scale(resolved_learner_scale, actor_value=resolved_actor_scale)


def format_attached_reference_policy_message(
    *,
    policy_id: str,
    coef: float,
    family_coef: float,
    raw_b1_distill_enabled: bool,
    weights_path: object,
) -> str:
    return (
        "Attached frozen reference policy: "
        f"policy_id={policy_id} coef={float(coef):g} family_coef={float(family_coef):g} "
        f"raw_b1_distill={bool(raw_b1_distill_enabled)} weights={weights_path}"
    )
