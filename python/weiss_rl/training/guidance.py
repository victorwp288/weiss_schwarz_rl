from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from weiss_rl.core.schedules import linear_anneal_value
from weiss_rl.models.loading import restore_model_guidance_from_payload as _restore_model_guidance_from_payload
from weiss_rl.runtime.components.teacher_labels import selected_teacher_label_profile

_TEACHER_LABEL_PROFILE_IDS = {"base": 0.0, "aggressive": 1.0, "control": 2.0}


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


def teacher_supervised_coef_scale_for_next_update(training_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=1.0,
            final_value=float(getattr(training_config, "teacher_supervised_final_scale", 1.0)),
            start_update=int(getattr(training_config, "teacher_supervised_start_updates", 0)),
            end_update=int(getattr(training_config, "teacher_supervised_end_updates", -1)),
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
        supervised_scale = teacher_supervised_coef_scale_for_next_update(
            training_config,
            update_count=update_count,
        )
        family_coef = float(getattr(training_config, "teacher_family_coef", 0.0)) * supervised_scale
        slot_coef = float(getattr(training_config, "teacher_slot_coef", 0.0)) * supervised_scale
        hand_coef = float(getattr(training_config, "teacher_hand_coef", 0.0)) * supervised_scale
        move_source_coef = float(getattr(training_config, "teacher_move_source_coef", 0.0)) * supervised_scale
        attack_type_coef = float(getattr(training_config, "teacher_attack_type_coef", 0.0)) * supervised_scale
        action_coef = float(getattr(training_config, "teacher_action_coef", 0.0)) * supervised_scale
        same_family_action_coef = (
            float(getattr(training_config, "teacher_same_family_action_coef", 0.0)) * supervised_scale
        )
        action_margin_coef = float(getattr(training_config, "teacher_action_margin_coef", 0.0)) * supervised_scale
        same_family_action_margin_coef = (
            float(getattr(training_config, "teacher_same_family_action_margin_coef", 0.0)) * supervised_scale
        )
        learner.set_teacher_aux_coefs(
            family=family_coef,
            slot=slot_coef,
            hand=hand_coef,
            move_source=move_source_coef,
            attack_type=attack_type_coef,
            action=action_coef,
            same_family_action=same_family_action_coef,
            action_margin=action_margin_coef,
            same_family_action_margin=same_family_action_margin_coef,
            public_heuristic=teacher_coef,
        )
        metrics["teacher_public_heuristic_coef_active"] = float(teacher_coef)
        metrics["teacher_supervised_coef_scale_active"] = float(supervised_scale)
        metrics["teacher_family_coef_active"] = float(family_coef)
        metrics["teacher_slot_coef_active"] = float(slot_coef)
        metrics["teacher_hand_coef_active"] = float(hand_coef)
        metrics["teacher_move_source_coef_active"] = float(move_source_coef)
        metrics["teacher_attack_type_coef_active"] = float(attack_type_coef)
        metrics["teacher_action_coef_active"] = float(action_coef)
        metrics["teacher_same_family_action_coef_active"] = float(same_family_action_coef)
        metrics["teacher_action_margin_coef_active"] = float(action_margin_coef)
        metrics["teacher_same_family_action_margin_coef_active"] = float(same_family_action_margin_coef)
        teacher_label_profile = selected_teacher_label_profile(
            getattr(training_config, "teacher_public_heuristic_profiles", ()),
            profile_mode=str(getattr(training_config, "teacher_public_heuristic_profile_mode", "mixture")),
            update_count=int(update_count),
            end_updates=int(getattr(training_config, "teacher_public_heuristic_profiles_end_updates", -1)),
        )
        metrics["teacher_label_profile_id_active"] = _TEACHER_LABEL_PROFILE_IDS[teacher_label_profile]
        for profile_name, profile_id in _TEACHER_LABEL_PROFILE_IDS.items():
            metrics[f"teacher_label_profile_{profile_name}_active"] = float(
                profile_id == _TEACHER_LABEL_PROFILE_IDS[teacher_label_profile]
            )
    model_config = stack.config.model
    if model is not None and model_config is not None:
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if callable(set_bias_scale):
            actor_bias_scale: float | None = None
            if callable(get_bias_scale):
                actor_bias_scale = float(get_bias_scale(scoring_mode="actor"))
            learner_bias_scale = public_heuristic_logit_bias_scale_for_next_update(
                model_config,
                update_count=update_count,
            )
            set_bias_scale(learner_bias_scale, actor_value=actor_bias_scale)
            metrics["public_heuristic_logit_bias_scale_active"] = float(learner_bias_scale)
            if actor_bias_scale is not None:
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
    payload: Mapping[str, Any],
) -> None:
    _restore_model_guidance_from_payload(model, payload)
