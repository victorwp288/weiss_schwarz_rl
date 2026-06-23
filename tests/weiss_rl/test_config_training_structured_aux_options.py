from __future__ import annotations

import pytest
from weiss_rl.config.sections_training import parse_training_config

from tests.weiss_rl.config_training_test_support import training_body


def test_parse_training_config_accepts_structured_public_heuristic_options() -> None:
    body = training_body()
    body["structured_aux"] = {
        "enabled": True,
        "teacher_public_heuristic_coef": 0.3,
        "teacher_public_heuristic_start_updates": 10,
        "teacher_public_heuristic_end_updates": 20,
        "teacher_public_heuristic_final_coef": 0.1,
        "teacher_public_heuristic_temperature": 24.0,
        "teacher_public_heuristic_families": ["attack", "main_move"],
        "teacher_public_heuristic_profiles": [" Aggressive ", "control"],
        "teacher_public_heuristic_profile_mode": "cycle",
        "teacher_public_heuristic_profiles_end_updates": 30,
        "teacher_action_margin_coef": 0.08,
        "teacher_action_margin": 0.75,
        "teacher_same_family_action_margin_coef": 0.04,
        "teacher_same_family_action_margin": 0.6,
        "teacher_exact_action_families": ["attack", "main_move"],
        "teacher_public_nonpass_over_pass_coef": 0.08,
        "teacher_public_nonpass_over_pass_margin": 0.25,
        "trajectory_retention_coef": 0.06,
        "trajectory_retention_policy_ids": [" seed_a ", "seed_b"],
        "trajectory_retention_sources": ["Champions", "warmup_snapshots"],
        "trajectory_bc_dataset_path": "runs/bc/dataset.npz",
        "trajectory_bc_every_updates": 2,
        "trajectory_bc_aux_updates": 3,
        "trajectory_bc_batch_episodes": 4,
        "trajectory_bc_seed": 123,
        "trajectory_bc_focus_source_labels": [" repair_a ", "repair_b"],
        "trajectory_bc_focus_fraction": 0.5,
        "trajectory_bc_teacher_action_coef": 0.25,
        "trajectory_bc_teacher_same_family_action_coef": 0.65,
        "paired_swing_conflict_filter": "history",
        "paired_swing_loss_scope": "episode_mean",
        "paired_swing_compare_to": "top_other",
    }
    body["structured_warmstart"] = {
        "enabled": True,
        "updates": 5,
        "teacher_public_heuristic_coef": 0.8,
        "teacher_public_heuristic_temperature": 12.0,
        "teacher_public_heuristic_families": ["main_play_character"],
        "teacher_public_heuristic_profiles": ["base", "aggressive"],
        "teacher_public_heuristic_profile_mode": "mixture",
        "teacher_public_heuristic_profiles_end_updates": 2,
    }

    config = parse_training_config(body)

    assert config.structured_aux.enabled is True
    assert config.structured_aux.teacher_public_heuristic_final_coef == pytest.approx(0.1)
    assert config.structured_aux.teacher_public_heuristic_profiles == (
        "aggressive",
        "control",
    )
    assert config.structured_aux.teacher_public_heuristic_profile_mode == "cycle"
    assert config.teacher_action_margin_coef == pytest.approx(0.08)
    assert config.teacher_action_margin == pytest.approx(0.75)
    assert config.teacher_same_family_action_margin_coef == pytest.approx(0.04)
    assert config.teacher_same_family_action_margin == pytest.approx(0.6)
    assert config.teacher_exact_action_families == ("attack", "main_move")
    assert config.teacher_public_nonpass_over_pass_coef == pytest.approx(0.08)
    assert config.teacher_public_nonpass_over_pass_margin == pytest.approx(0.25)
    assert config.trajectory_retention_coef == pytest.approx(0.06)
    assert config.trajectory_retention_policy_ids == ("seed_a", "seed_b")
    assert config.trajectory_retention_sources == ("champions", "warmup_snapshots")
    assert config.trajectory_bc_enabled is True
    assert config.trajectory_bc_dataset_path == "runs/bc/dataset.npz"
    assert config.structured_aux.trajectory_bc_every_updates == 2
    assert config.structured_aux.trajectory_bc_aux_updates == 3
    assert config.structured_aux.trajectory_bc_batch_episodes == 4
    assert config.structured_aux.trajectory_bc_seed == 123
    assert config.structured_aux.trajectory_bc_focus_source_labels == (
        "repair_a",
        "repair_b",
    )
    assert config.structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.5)
    assert config.structured_aux.trajectory_bc_focus_groups == ()
    assert config.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.25)
    assert config.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.65)
    assert config.structured_aux.paired_swing_conflict_filter == "history"
    assert config.structured_aux.paired_swing_loss_scope == "episode_mean"
    assert config.structured_aux.paired_swing_compare_to == "top_other"
    assert config.structured_warmstart.enabled is True
    assert config.structured_warmstart.updates == 5
    assert config.structured_warmstart.teacher_public_heuristic_profiles == (
        "base",
        "aggressive",
    )


def test_parse_training_config_reuses_public_heuristic_coef_as_final_default() -> None:
    body = training_body()
    body["structured_aux"] = {"teacher_public_heuristic_coef": 0.25}

    config = parse_training_config(body)

    assert config.structured_aux.teacher_public_heuristic_final_coef == pytest.approx(0.25)
