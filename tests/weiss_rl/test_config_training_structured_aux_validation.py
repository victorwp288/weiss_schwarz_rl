from __future__ import annotations

import pytest
from weiss_rl.config.sections.sections_training import parse_training_config

from tests.weiss_rl.config_training_test_support import training_body


def test_parse_training_config_preserves_structured_public_heuristic_errors() -> None:
    bad_temp = training_body()
    bad_temp["structured_aux"] = {"teacher_public_heuristic_temperature": 0.0}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_heuristic_temperature must be > 0",
    ):
        parse_training_config(bad_temp)

    bad_profile = training_body()
    bad_profile["structured_aux"] = {"teacher_public_heuristic_profiles": ["base", "unknown"]}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_heuristic_profiles contains unsupported profiles: unknown",
    ):
        parse_training_config(bad_profile)

    bad_profile_mode = training_body()
    bad_profile_mode["structured_warmstart"] = {"teacher_public_heuristic_profile_mode": "round_robin"}
    with pytest.raises(
        ValueError,
        match="training.structured_warmstart.teacher_public_heuristic_profile_mode must be one of: cycle, mixture",
    ):
        parse_training_config(bad_profile_mode)

    bad_final = training_body()
    bad_final["structured_aux"] = {"teacher_public_heuristic_final_coef": -0.01}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_heuristic_final_coef must be >= 0.0",
    ):
        parse_training_config(bad_final)

    bad_margin_coef = training_body()
    bad_margin_coef["structured_aux"] = {"teacher_action_margin_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_action_margin_coef must be >= 0.0",
    ):
        parse_training_config(bad_margin_coef)

    bad_margin = training_body()
    bad_margin["structured_aux"] = {"teacher_action_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_action_margin must be >= 0.0",
    ):
        parse_training_config(bad_margin)

    bad_same_family_margin_coef = training_body()
    bad_same_family_margin_coef["structured_aux"] = {"teacher_same_family_action_margin_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_same_family_action_margin_coef must be >= 0.0",
    ):
        parse_training_config(bad_same_family_margin_coef)

    bad_same_family_margin = training_body()
    bad_same_family_margin["structured_aux"] = {"teacher_same_family_action_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_same_family_action_margin must be >= 0.0",
    ):
        parse_training_config(bad_same_family_margin)

    bad_nonpass_coef = training_body()
    bad_nonpass_coef["structured_aux"] = {"teacher_public_nonpass_over_pass_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_nonpass_over_pass_coef must be >= 0.0",
    ):
        parse_training_config(bad_nonpass_coef)

    bad_nonpass_margin = training_body()
    bad_nonpass_margin["structured_aux"] = {"teacher_public_nonpass_over_pass_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_nonpass_over_pass_margin must be >= 0.0",
    ):
        parse_training_config(bad_nonpass_margin)

    bad_anchor_coef = training_body()
    bad_anchor_coef["structured_aux"] = {"policy_anchor_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_coef must be >= 0.0",
    ):
        parse_training_config(bad_anchor_coef)

    bad_anchor_top_action_coef = training_body()
    bad_anchor_top_action_coef["structured_aux"] = {"policy_anchor_top_action_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_top_action_coef must be >= 0.0",
    ):
        parse_training_config(bad_anchor_top_action_coef)

    bad_anchor_temperature = training_body()
    bad_anchor_temperature["structured_aux"] = {"policy_anchor_temperature": 0.0}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_temperature must be > 0",
    ):
        parse_training_config(bad_anchor_temperature)

    bad_retention_coef = training_body()
    bad_retention_coef["structured_aux"] = {"trajectory_retention_coef": -0.01}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.trajectory_retention_coef must be >= 0.0",
    ):
        parse_training_config(bad_retention_coef)

    bad_retention_source = training_body()
    bad_retention_source["structured_aux"] = {"trajectory_retention_sources": ["champions", "unknown"]}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.trajectory_retention_sources contains unsupported sources: unknown",
    ):
        parse_training_config(bad_retention_source)

    bad_focus_fraction = training_body()
    bad_focus_fraction["structured_aux"] = {"trajectory_bc_focus_fraction": 1.25}
    with pytest.raises(ValueError, match="trajectory_bc_focus_fraction must be between"):
        parse_training_config(bad_focus_fraction)

    bad_focus_contract = training_body()
    bad_focus_contract["structured_aux"] = {
        "trajectory_bc_focus_source_labels": ["repair_a"],
        "trajectory_bc_focus_fraction": 0.25,
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_b"], "fraction": 0.25},
        ],
    }
    with pytest.raises(ValueError, match="trajectory_bc_focus_groups cannot be combined"):
        parse_training_config(bad_focus_contract)

    bad_paired_swing_scope = training_body()
    bad_paired_swing_scope["structured_aux"] = {"paired_swing_loss_scope": "trajectory"}
    with pytest.raises(ValueError, match="paired_swing_loss_scope"):
        parse_training_config(bad_paired_swing_scope)

    bad_focus_overlap = training_body()
    bad_focus_overlap["structured_aux"] = {
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_a"], "fraction": 0.25},
            {"name": "fixed", "source_labels": ["repair_a"], "fraction": 0.25},
        ],
    }
    with pytest.raises(ValueError, match="contains labels in multiple groups: repair_a"):
        parse_training_config(bad_focus_overlap)

    bad_focus_sum = training_body()
    bad_focus_sum["structured_aux"] = {
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_a"], "fraction": 0.70},
            {"name": "fixed", "source_labels": ["repair_b"], "fraction": 0.40},
        ],
    }
    with pytest.raises(ValueError, match="fractions must sum to <= 1.0"):
        parse_training_config(bad_focus_sum)
