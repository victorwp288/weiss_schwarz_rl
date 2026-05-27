"""Evaluation stack config section parser."""

from __future__ import annotations

from typing import Any

from .models import (
    DecisionKindTaggingConfig,
    EvaluationConfig,
    FinalPolicySetSelectionConfig,
    FixedAnchorSetConfig,
    LegalFingerprintChecksConfig,
    StopRulesConfig,
)
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_float,
    require_int,
    require_int_list,
    require_mapping,
    require_str_list,
    require_text,
)


def parse_evaluation_config(body: dict[str, Any]) -> EvaluationConfig:
    reject_unknown_keys(
        body,
        allowed={
            "seat_swap",
            "eval_device",
            "eval_inference_mode",
            "eval_sampling_algorithm",
            "model_sampling_temperature",
            "eval_assert_sorted_legal_ids",
            "seed_files",
            "periodic_dev_eval_interval_updates",
            "periodic_dev_eval_paired_seeds",
            "final_policy_set_size",
            "final_matrix_stage1_paired_seeds",
            "final_matrix_stage2_adaptive_max_paired_seeds",
            "stop_rules",
            "replay_capture_rate_eval",
            "regression_capture_count",
            "legal_fingerprint_checks",
            "decision_kind_tagging",
            "final_policy_set_selection",
        },
        context="evaluation",
    )
    seed_files = require_mapping(body["seed_files"], context="evaluation.seed_files")
    stop_rules = require_mapping(body["stop_rules"], context="evaluation.stop_rules")
    legal = require_mapping(body["legal_fingerprint_checks"], context="evaluation.legal_fingerprint_checks")
    decision = require_mapping(body["decision_kind_tagging"], context="evaluation.decision_kind_tagging")
    selection = require_mapping(body["final_policy_set_selection"], context="evaluation.final_policy_set_selection")
    fixed_anchor = require_mapping(
        selection["fixed_anchor_set_v1"], context="evaluation.final_policy_set_selection.fixed_anchor_set_v1"
    )
    reject_unknown_keys(
        stop_rules, allowed={"stop_delta_ci_half_width", "stop_confidence"}, context="evaluation.stop_rules"
    )
    reject_unknown_keys(
        legal,
        allowed={"enabled", "version", "require_strictly_increasing_legal_ids", "mismatch_policy"},
        context="evaluation.legal_fingerprint_checks",
    )
    reject_unknown_keys(
        decision,
        allowed={"required_for_training", "enable_python_derived_debug_tag"},
        context="evaluation.decision_kind_tagging",
    )
    reject_unknown_keys(
        selection,
        allowed={
            "version",
            "include_random_legal_baseline_b0",
            "include_no_league_baseline_b1",
            "include_heuristic_public_b2_if_exists",
            "include_heuristic_public_anchors_b2_b3_b4",
            "include_final_champion_snapshot",
            "include_spaced_snapshots_near_percent_updates",
            "remaining_slots_strategy",
            "fixed_anchor_set_v1",
            "seed_file",
            "folding",
            "seat_swap",
            "tie_break",
        },
        context="evaluation.final_policy_set_selection",
    )
    reject_unknown_keys(
        fixed_anchor,
        allowed={"required", "optional_if_available"},
        context="evaluation.final_policy_set_selection.fixed_anchor_set_v1",
    )
    mismatch_policy = require_text(
        legal["mismatch_policy"],
        field_name="evaluation.legal_fingerprint_checks.mismatch_policy",
    )
    if mismatch_policy != "hard_fail":
        raise ValueError(
            f"evaluation.legal_fingerprint_checks.mismatch_policy must be 'hard_fail', got {mismatch_policy!r}"
        )
    model_sampling_temperature = require_float(
        body.get("model_sampling_temperature", 1.0),
        field_name="evaluation.model_sampling_temperature",
    )
    if model_sampling_temperature <= 0.0:
        raise ValueError(f"evaluation.model_sampling_temperature must be > 0, got {model_sampling_temperature!r}")
    return EvaluationConfig(
        seat_swap=require_bool(body["seat_swap"], field_name="evaluation.seat_swap"),
        eval_device=require_text(body["eval_device"], field_name="evaluation.eval_device"),
        eval_inference_mode=require_bool(body["eval_inference_mode"], field_name="evaluation.eval_inference_mode"),
        eval_sampling_algorithm=require_text(
            body["eval_sampling_algorithm"],
            field_name="evaluation.eval_sampling_algorithm",
        ),
        eval_assert_sorted_legal_ids=require_bool(
            body["eval_assert_sorted_legal_ids"],
            field_name="evaluation.eval_assert_sorted_legal_ids",
        ),
        seed_files={
            key: require_text(value, field_name=f"evaluation.seed_files.{key}") for key, value in seed_files.items()
        },
        periodic_dev_eval_interval_updates=require_int(
            body["periodic_dev_eval_interval_updates"],
            field_name="evaluation.periodic_dev_eval_interval_updates",
            minimum=0,
        ),
        periodic_dev_eval_paired_seeds=require_int(
            body["periodic_dev_eval_paired_seeds"],
            field_name="evaluation.periodic_dev_eval_paired_seeds",
            minimum=1,
        ),
        final_policy_set_size=require_int(
            body["final_policy_set_size"], field_name="evaluation.final_policy_set_size", minimum=1
        ),
        final_matrix_stage1_paired_seeds=require_int(
            body["final_matrix_stage1_paired_seeds"],
            field_name="evaluation.final_matrix_stage1_paired_seeds",
            minimum=1,
        ),
        final_matrix_stage2_adaptive_max_paired_seeds=require_int(
            body["final_matrix_stage2_adaptive_max_paired_seeds"],
            field_name="evaluation.final_matrix_stage2_adaptive_max_paired_seeds",
            minimum=1,
        ),
        stop_rules=StopRulesConfig(
            stop_delta_ci_half_width=require_float(
                stop_rules["stop_delta_ci_half_width"],
                field_name="evaluation.stop_rules.stop_delta_ci_half_width",
            ),
            stop_confidence=require_float(
                stop_rules["stop_confidence"],
                field_name="evaluation.stop_rules.stop_confidence",
            ),
        ),
        replay_capture_rate_eval=require_float(
            body["replay_capture_rate_eval"],
            field_name="evaluation.replay_capture_rate_eval",
        ),
        regression_capture_count=require_int(
            body["regression_capture_count"],
            field_name="evaluation.regression_capture_count",
            minimum=0,
        ),
        legal_fingerprint_checks=LegalFingerprintChecksConfig(
            enabled=require_bool(legal["enabled"], field_name="evaluation.legal_fingerprint_checks.enabled"),
            version=require_text(legal["version"], field_name="evaluation.legal_fingerprint_checks.version"),
            require_strictly_increasing_legal_ids=require_bool(
                legal["require_strictly_increasing_legal_ids"],
                field_name="evaluation.legal_fingerprint_checks.require_strictly_increasing_legal_ids",
            ),
            mismatch_policy=require_text(
                mismatch_policy,
                field_name="evaluation.legal_fingerprint_checks.mismatch_policy",
            ),
        ),
        decision_kind_tagging=DecisionKindTaggingConfig(
            required_for_training=require_bool(
                decision["required_for_training"],
                field_name="evaluation.decision_kind_tagging.required_for_training",
            ),
            enable_python_derived_debug_tag=require_bool(
                decision["enable_python_derived_debug_tag"],
                field_name="evaluation.decision_kind_tagging.enable_python_derived_debug_tag",
            ),
        ),
        final_policy_set_selection=FinalPolicySetSelectionConfig(
            version=require_text(selection["version"], field_name="evaluation.final_policy_set_selection.version"),
            include_random_legal_baseline_b0=require_bool(
                selection["include_random_legal_baseline_b0"],
                field_name="evaluation.final_policy_set_selection.include_random_legal_baseline_b0",
            ),
            include_no_league_baseline_b1=require_bool(
                selection["include_no_league_baseline_b1"],
                field_name="evaluation.final_policy_set_selection.include_no_league_baseline_b1",
            ),
            include_heuristic_public_b2_if_exists=require_bool(
                selection["include_heuristic_public_b2_if_exists"],
                field_name="evaluation.final_policy_set_selection.include_heuristic_public_b2_if_exists",
            ),
            include_heuristic_public_anchors_b2_b3_b4=require_bool(
                selection.get("include_heuristic_public_anchors_b2_b3_b4", False),
                field_name="evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4",
            ),
            include_final_champion_snapshot=require_bool(
                selection["include_final_champion_snapshot"],
                field_name="evaluation.final_policy_set_selection.include_final_champion_snapshot",
            ),
            include_spaced_snapshots_near_percent_updates=require_int_list(
                selection["include_spaced_snapshots_near_percent_updates"],
                field_name="evaluation.final_policy_set_selection.include_spaced_snapshots_near_percent_updates",
            ),
            remaining_slots_strategy=require_text(
                selection["remaining_slots_strategy"],
                field_name="evaluation.final_policy_set_selection.remaining_slots_strategy",
            ),
            fixed_anchor_set_v1=FixedAnchorSetConfig(
                required=require_str_list(
                    fixed_anchor["required"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.required",
                ),
                optional_if_available=require_str_list(
                    fixed_anchor["optional_if_available"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available",
                ),
            ),
            seed_file=require_text(
                selection["seed_file"], field_name="evaluation.final_policy_set_selection.seed_file"
            ),
            folding=require_text(selection["folding"], field_name="evaluation.final_policy_set_selection.folding"),
            seat_swap=require_bool(
                selection["seat_swap"], field_name="evaluation.final_policy_set_selection.seat_swap"
            ),
            tie_break=require_text(
                selection["tie_break"], field_name="evaluation.final_policy_set_selection.tie_break"
            ),
        ),
        model_sampling_temperature=model_sampling_temperature,
    )
