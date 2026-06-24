"""Structured auxiliary and warmstart config parsing."""

from __future__ import annotations

from typing import Any

from weiss_rl.config.loading.parsing_utils import require_bool, require_float, require_int, require_str_list
from weiss_rl.config.models import TrainingStructuredAuxConfig, TrainingStructuredWarmstartConfig

from .sections_training_aux_helpers import (
    require_nonnegative_float,
    require_positive_float,
    require_update_window,
    trajectory_retention_sources,
)
from .sections_training_public_teacher import parse_public_heuristic_teacher_profile_settings
from .sections_training_replay_aux import parse_structured_aux_replay_data_settings


def parse_training_structured_aux_config(structured_aux: dict[str, Any]) -> TrainingStructuredAuxConfig:
    public_profile_settings = parse_public_heuristic_teacher_profile_settings(
        structured_aux,
        context="training.structured_aux",
    )

    action_margin_coef = require_nonnegative_float(
        structured_aux,
        "teacher_action_margin_coef",
        0.0,
        field_name="training.structured_aux.teacher_action_margin_coef",
    )
    action_margin = require_nonnegative_float(
        structured_aux,
        "teacher_action_margin",
        0.5,
        field_name="training.structured_aux.teacher_action_margin",
    )

    same_family_action_margin_coef = require_nonnegative_float(
        structured_aux,
        "teacher_same_family_action_margin_coef",
        0.0,
        field_name="training.structured_aux.teacher_same_family_action_margin_coef",
    )
    same_family_action_margin = require_nonnegative_float(
        structured_aux,
        "teacher_same_family_action_margin",
        0.5,
        field_name="training.structured_aux.teacher_same_family_action_margin",
    )

    supervised_start_updates, supervised_end_updates = require_update_window(
        structured_aux,
        start_key="teacher_supervised_start_updates",
        end_key="teacher_supervised_end_updates",
        context="training.structured_aux",
    )
    supervised_final_scale = require_nonnegative_float(
        structured_aux,
        "teacher_supervised_final_scale",
        1.0,
        field_name="training.structured_aux.teacher_supervised_final_scale",
    )

    public_nonpass_coef = require_nonnegative_float(
        structured_aux,
        "teacher_public_nonpass_over_pass_coef",
        0.0,
        field_name="training.structured_aux.teacher_public_nonpass_over_pass_coef",
    )
    public_nonpass_margin = require_nonnegative_float(
        structured_aux,
        "teacher_public_nonpass_over_pass_margin",
        0.5,
        field_name="training.structured_aux.teacher_public_nonpass_over_pass_margin",
    )

    public_start_updates, public_end_updates = require_update_window(
        structured_aux,
        start_key="teacher_public_heuristic_start_updates",
        end_key="teacher_public_heuristic_end_updates",
        context="training.structured_aux",
    )
    public_final_coef = require_float(
        structured_aux.get(
            "teacher_public_heuristic_final_coef",
            structured_aux.get("teacher_public_heuristic_coef", 0.0),
        ),
        field_name="training.structured_aux.teacher_public_heuristic_final_coef",
    )
    if public_final_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_public_heuristic_final_coef must be >= 0.0")
    policy_anchor_coef = require_nonnegative_float(
        structured_aux,
        "policy_anchor_coef",
        0.0,
        field_name="training.structured_aux.policy_anchor_coef",
    )
    policy_anchor_top_action_coef = require_nonnegative_float(
        structured_aux,
        "policy_anchor_top_action_coef",
        0.0,
        field_name="training.structured_aux.policy_anchor_top_action_coef",
    )
    policy_anchor_temperature = require_positive_float(
        structured_aux,
        "policy_anchor_temperature",
        1.0,
        field_name="training.structured_aux.policy_anchor_temperature",
    )

    trajectory_retention_coef = require_nonnegative_float(
        structured_aux,
        "trajectory_retention_coef",
        0.0,
        field_name="training.structured_aux.trajectory_retention_coef",
    )
    trajectory_retention_policy_ids = tuple(
        str(policy_id).strip()
        for policy_id in require_str_list(
            structured_aux.get("trajectory_retention_policy_ids", []),
            field_name="training.structured_aux.trajectory_retention_policy_ids",
        )
        if str(policy_id).strip()
    )
    retention_sources = trajectory_retention_sources(structured_aux)
    replay_data_settings = parse_structured_aux_replay_data_settings(structured_aux)

    return TrainingStructuredAuxConfig(
        enabled=require_bool(
            structured_aux.get("enabled", False),
            field_name="training.structured_aux.enabled",
        ),
        teacher_family_coef=require_float(
            structured_aux.get("teacher_family_coef", 0.0),
            field_name="training.structured_aux.teacher_family_coef",
        ),
        teacher_slot_coef=require_float(
            structured_aux.get("teacher_slot_coef", 0.0),
            field_name="training.structured_aux.teacher_slot_coef",
        ),
        teacher_hand_coef=require_float(
            structured_aux.get("teacher_hand_coef", 0.0),
            field_name="training.structured_aux.teacher_hand_coef",
        ),
        teacher_move_source_coef=require_float(
            structured_aux.get("teacher_move_source_coef", 0.0),
            field_name="training.structured_aux.teacher_move_source_coef",
        ),
        teacher_attack_type_coef=require_float(
            structured_aux.get("teacher_attack_type_coef", 0.0),
            field_name="training.structured_aux.teacher_attack_type_coef",
        ),
        teacher_action_coef=require_float(
            structured_aux.get("teacher_action_coef", 0.0),
            field_name="training.structured_aux.teacher_action_coef",
        ),
        teacher_same_family_action_coef=require_float(
            structured_aux.get("teacher_same_family_action_coef", 0.0),
            field_name="training.structured_aux.teacher_same_family_action_coef",
        ),
        teacher_action_margin_coef=action_margin_coef,
        teacher_action_margin=action_margin,
        teacher_same_family_action_margin_coef=same_family_action_margin_coef,
        teacher_same_family_action_margin=same_family_action_margin,
        teacher_supervised_start_updates=supervised_start_updates,
        teacher_supervised_end_updates=supervised_end_updates,
        teacher_supervised_final_scale=supervised_final_scale,
        teacher_exact_action_families=require_str_list(
            structured_aux.get("teacher_exact_action_families", []),
            field_name="training.structured_aux.teacher_exact_action_families",
        ),
        teacher_public_heuristic_coef=require_float(
            structured_aux.get("teacher_public_heuristic_coef", 0.0),
            field_name="training.structured_aux.teacher_public_heuristic_coef",
        ),
        teacher_public_heuristic_start_updates=public_start_updates,
        teacher_public_heuristic_end_updates=public_end_updates,
        teacher_public_heuristic_final_coef=require_float(
            public_final_coef,
            field_name="training.structured_aux.teacher_public_heuristic_final_coef",
        ),
        teacher_public_heuristic_temperature=public_profile_settings.temperature,
        teacher_public_nonpass_over_pass_coef=public_nonpass_coef,
        teacher_public_nonpass_over_pass_margin=public_nonpass_margin,
        teacher_public_heuristic_families=require_str_list(
            structured_aux.get("teacher_public_heuristic_families", []),
            field_name="training.structured_aux.teacher_public_heuristic_families",
        ),
        teacher_public_heuristic_profiles=public_profile_settings.profiles,
        teacher_public_heuristic_profile_mode=public_profile_settings.profile_mode,
        teacher_public_heuristic_profiles_end_updates=public_profile_settings.profiles_end_updates,
        policy_anchor_coef=policy_anchor_coef,
        policy_anchor_top_action_coef=policy_anchor_top_action_coef,
        policy_anchor_temperature=policy_anchor_temperature,
        trajectory_retention_coef=trajectory_retention_coef,
        trajectory_retention_policy_ids=trajectory_retention_policy_ids,
        trajectory_retention_sources=retention_sources,
        **replay_data_settings,
    )


def parse_training_structured_warmstart_config(
    structured_warmstart: dict[str, Any],
) -> TrainingStructuredWarmstartConfig:
    public_profile_settings = parse_public_heuristic_teacher_profile_settings(
        structured_warmstart,
        context="training.structured_warmstart",
    )

    return TrainingStructuredWarmstartConfig(
        enabled=require_bool(
            structured_warmstart.get("enabled", False),
            field_name="training.structured_warmstart.enabled",
        ),
        updates=require_int(
            structured_warmstart.get("updates", 0),
            field_name="training.structured_warmstart.updates",
            minimum=0,
        ),
        teacher_family_coef=require_float(
            structured_warmstart.get("teacher_family_coef", 0.0),
            field_name="training.structured_warmstart.teacher_family_coef",
        ),
        teacher_slot_coef=require_float(
            structured_warmstart.get("teacher_slot_coef", 0.0),
            field_name="training.structured_warmstart.teacher_slot_coef",
        ),
        teacher_hand_coef=require_float(
            structured_warmstart.get("teacher_hand_coef", 0.0),
            field_name="training.structured_warmstart.teacher_hand_coef",
        ),
        teacher_move_source_coef=require_float(
            structured_warmstart.get("teacher_move_source_coef", 0.0),
            field_name="training.structured_warmstart.teacher_move_source_coef",
        ),
        teacher_attack_type_coef=require_float(
            structured_warmstart.get("teacher_attack_type_coef", 0.0),
            field_name="training.structured_warmstart.teacher_attack_type_coef",
        ),
        teacher_action_coef=require_float(
            structured_warmstart.get("teacher_action_coef", 0.0),
            field_name="training.structured_warmstart.teacher_action_coef",
        ),
        teacher_same_family_action_coef=require_float(
            structured_warmstart.get("teacher_same_family_action_coef", 0.0),
            field_name="training.structured_warmstart.teacher_same_family_action_coef",
        ),
        teacher_public_heuristic_coef=require_float(
            structured_warmstart.get("teacher_public_heuristic_coef", 0.0),
            field_name="training.structured_warmstart.teacher_public_heuristic_coef",
        ),
        teacher_public_heuristic_temperature=public_profile_settings.temperature,
        teacher_public_heuristic_families=require_str_list(
            structured_warmstart.get("teacher_public_heuristic_families", []),
            field_name="training.structured_warmstart.teacher_public_heuristic_families",
        ),
        teacher_public_heuristic_profiles=public_profile_settings.profiles,
        teacher_public_heuristic_profile_mode=public_profile_settings.profile_mode,
        teacher_public_heuristic_profiles_end_updates=public_profile_settings.profiles_end_updates,
    )


__all__ = [
    "parse_training_structured_aux_config",
    "parse_training_structured_warmstart_config",
]
