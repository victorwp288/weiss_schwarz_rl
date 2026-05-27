from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from weiss_rl.config.sections_training import parse_training_config


def _copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def _training_body() -> dict[str, object]:
    return {
        "algorithm": "impala_vtrace_gru",
        "rollout": {"unroll_length": 4, "batch_unrolls_per_update": 2},
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "grad_norm_clip": 0.5,
            "value_loss_coef": 0.25,
        },
        "exploration": {
            "entropy_coef": 0.01,
            "entropy_anneal_to": 0.001,
            "entropy_anneal_steps_updates": 100,
        },
        "precision": {
            "mixed_precision": False,
            "compile_learner": False,
            "masking_math_float32": True,
        },
        "checkpointing": {
            "checkpoint_interval_updates": 10,
            "snapshot_interval_updates": 20,
            "actor_reload_interval_updates": 5,
        },
        "vtrace": {"rho_bar": 1.0, "c_bar": 1.0},
    }


def test_parse_training_config_applies_existing_defaults() -> None:
    config = parse_training_config(_training_body())

    assert config.algorithm == "impala_vtrace_gru"
    assert config.compile_actor_inference is False
    assert config.profile_timers is False
    assert config.torch_profiler is False
    assert config.ppo_clip_epsilon == pytest.approx(0.2)
    assert config.ppo_value_clip_epsilon == pytest.approx(0.2)
    assert config.ppo_gae_lambda == pytest.approx(0.95)
    assert config.ppo_epochs == 4
    assert config.ppo_target_kl == pytest.approx(0.0)
    assert config.ppo_normalize_advantages is True
    assert config.structured_aux_enabled is False
    assert config.structured_metrics_mode == "off"
    assert config.teacher_aux_mode == "always"
    assert config.fixed_opponent_backend == "python_scalar"
    assert config.fixed_model_opponent_action_selection == "sample"
    assert config.actor_policy_backend == "model"
    assert config.actor_heuristic_fraction == pytest.approx(1.0)
    assert config.actor_heuristic_end_updates == -1
    assert config.actor_heuristic_final_fraction == pytest.approx(1.0)
    assert config.train_on_heuristic_actor_rows is True
    assert config.heuristic_actor_hidden_state_tracking is True
    assert config.diverse_opponent_actor_count == 0
    assert config.diverse_opponent_batch_fraction == pytest.approx(0.0)
    assert config.entropy_scope == "candidate"
    assert config.actor_sampling_temperature == pytest.approx(1.0)
    assert config.mulligan_force_confirm_after_select is False
    assert config.force_pass_over_main_move_only is False
    assert config.main_move_only_max_consecutive == 0
    assert config.force_attack_over_pass_when_attack_legal is False
    assert config.teacher_action_margin_coef == pytest.approx(0.0)
    assert config.teacher_action_margin == pytest.approx(0.5)
    assert config.teacher_same_family_action_margin_coef == pytest.approx(0.0)
    assert config.teacher_same_family_action_margin == pytest.approx(0.5)
    assert config.teacher_exact_action_families == ()
    assert config.teacher_public_nonpass_over_pass_coef == pytest.approx(0.0)
    assert config.teacher_public_nonpass_over_pass_margin == pytest.approx(0.5)
    assert config.trajectory_retention_coef == pytest.approx(0.0)
    assert config.trajectory_retention_policy_ids == ()
    assert config.trajectory_retention_sources == ("champions",)
    assert config.trajectory_bc_enabled is False
    assert config.trajectory_bc_dataset_path == ""
    assert config.structured_aux.trajectory_bc_every_updates == 0


def test_parse_training_config_accepts_nested_overrides() -> None:
    body = _training_body()
    body.update(
        {
            "algorithm": "ppo_lite_masked_v1",
            "profile_timers": True,
            "torch_profiler": True,
            "fixed_opponent_backend": "python_batched",
            "fixed_model_opponent_action_selection": "argmax",
            "actor_policy_backend": "heuristic_public",
            "actor_heuristic_fraction": 0.75,
            "actor_heuristic_start_updates": 10,
            "actor_heuristic_end_updates": 20,
            "actor_heuristic_final_fraction": 0.25,
            "train_on_heuristic_actor_rows": False,
            "diverse_opponent_actor_count": 4,
            "diverse_model_actor_count": 2,
            "diverse_opponent_batch_fraction": 0.125,
            "diverse_opponent_batch_wait_ms": 50,
            "heuristic_actor_hidden_state_tracking": False,
            "ppo": {
                "clip_epsilon": 0.3,
                "value_clip_epsilon": 0.4,
                "gae_lambda": 0.8,
                "epochs": 3,
                "target_kl": 0.02,
                "normalize_advantages": False,
            },
            "structured_metrics": {"mode": "sampled"},
            "teacher_aux": {"mode": "warmstart_only"},
            "action_surface": {
                "mulligan_force_confirm_after_select": True,
                "force_pass_over_main_move_only": True,
                "main_move_only_max_consecutive": 1,
                "force_attack_over_pass_when_attack_legal": True,
            },
        }
    )
    precision = _copy_section(body, "precision")
    precision["compile_actor_inference"] = True
    body["precision"] = precision
    exploration = _copy_section(body, "exploration")
    exploration["entropy_scope"] = "family"
    exploration["actor_sampling_temperature"] = 0.25
    body["exploration"] = exploration

    config = parse_training_config(body)

    assert config.algorithm == "ppo_lite_masked_v1"
    assert config.compile_actor_inference is True
    assert config.profile_timers is True
    assert config.torch_profiler is True
    assert config.fixed_opponent_backend == "python_batched"
    assert config.fixed_model_opponent_action_selection == "argmax"
    assert config.actor_policy_backend == "heuristic_public"
    assert config.actor_heuristic_final_fraction == pytest.approx(0.25)
    assert config.train_on_heuristic_actor_rows is False
    assert config.diverse_opponent_actor_count == 4
    assert config.diverse_model_actor_count == 2
    assert config.diverse_opponent_batch_fraction == pytest.approx(0.125)
    assert config.diverse_opponent_batch_wait_ms == 50
    assert config.heuristic_actor_hidden_state_tracking is False
    assert config.ppo_clip_epsilon == pytest.approx(0.3)
    assert config.ppo_normalize_advantages is False
    assert config.structured_metrics_mode == "sampled"
    assert config.teacher_aux_mode == "warmstart_only"
    assert config.entropy_scope == "family"
    assert config.actor_sampling_temperature == pytest.approx(0.25)
    assert config.mulligan_force_confirm_after_select is True
    assert config.force_pass_over_main_move_only is True
    assert config.main_move_only_max_consecutive == 1
    assert config.force_attack_over_pass_when_attack_legal is True


def test_parse_training_config_accepts_structured_public_heuristic_options() -> None:
    body = _training_body()
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
    assert config.structured_aux.teacher_public_heuristic_profiles == ("aggressive", "control")
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
    assert config.structured_aux.trajectory_bc_focus_source_labels == ("repair_a", "repair_b")
    assert config.structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.5)
    assert config.structured_aux.trajectory_bc_focus_groups == ()
    assert config.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.25)
    assert config.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.65)
    assert config.structured_aux.paired_swing_conflict_filter == "history"
    assert config.structured_aux.paired_swing_loss_scope == "episode_mean"
    assert config.structured_aux.paired_swing_compare_to == "top_other"
    assert config.structured_warmstart.enabled is True
    assert config.structured_warmstart.updates == 5
    assert config.structured_warmstart.teacher_public_heuristic_profiles == ("base", "aggressive")


def test_parse_training_config_accepts_trajectory_bc_focus_groups() -> None:
    body = _training_body()
    body["structured_aux"] = {
        "trajectory_bc_dataset_path": "runs/bc/dataset.npz",
        "trajectory_bc_every_updates": 1,
        "trajectory_bc_focus_groups": [
            {
                "name": " learned_repair ",
                "source_labels": [" champion_a ", "hard_negative_a"],
                "fraction": 0.30,
            },
            {
                "name": "fixed_repair",
                "source_labels": ["b1_lossstate", "b3_lossstate"],
                "fraction": 0.25,
            },
        ],
    }

    config = parse_training_config(body)

    assert config.trajectory_bc_enabled is True
    assert config.structured_aux.trajectory_bc_focus_source_labels == ()
    assert config.structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.0)
    assert tuple(group.name for group in config.structured_aux.trajectory_bc_focus_groups) == (
        "learned_repair",
        "fixed_repair",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "champion_a",
        "hard_negative_a",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[0].fraction == pytest.approx(0.30)
    assert config.structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "b1_lossstate",
        "b3_lossstate",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[1].fraction == pytest.approx(0.25)


def test_parse_training_config_reuses_public_heuristic_coef_as_final_default() -> None:
    body = _training_body()
    body["structured_aux"] = {"teacher_public_heuristic_coef": 0.25}

    config = parse_training_config(body)

    assert config.structured_aux.teacher_public_heuristic_final_coef == pytest.approx(0.25)


def test_parse_training_config_preserves_choice_and_range_errors() -> None:
    bad_algorithm = _training_body()
    bad_algorithm["algorithm"] = "new_algo"
    with pytest.raises(ValueError, match="training.algorithm must be one of:"):
        parse_training_config(bad_algorithm)

    bad_fraction = _training_body()
    bad_fraction["actor_heuristic_fraction"] = 1.5
    with pytest.raises(
        ValueError,
        match="training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive, got 1.5",
    ):
        parse_training_config(bad_fraction)

    bad_schedule = _training_body()
    bad_schedule["actor_heuristic_start_updates"] = 10
    bad_schedule["actor_heuristic_end_updates"] = 5
    with pytest.raises(
        ValueError,
        match="training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates",
    ):
        parse_training_config(bad_schedule)

    bad_diverse_fraction = _training_body()
    bad_diverse_fraction["diverse_opponent_batch_fraction"] = -0.1
    with pytest.raises(
        ValueError,
        match="training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive, got -0.1",
    ):
        parse_training_config(bad_diverse_fraction)

    bad_temperature = _training_body()
    exploration = _copy_section(bad_temperature, "exploration")
    exploration["actor_sampling_temperature"] = 0.0
    bad_temperature["exploration"] = exploration
    with pytest.raises(ValueError, match="actor_sampling_temperature must be > 0"):
        parse_training_config(bad_temperature)


def test_parse_training_config_preserves_structured_public_heuristic_errors() -> None:
    bad_temp = _training_body()
    bad_temp["structured_aux"] = {"teacher_public_heuristic_temperature": 0.0}
    with pytest.raises(ValueError, match="training.structured_aux.teacher_public_heuristic_temperature must be > 0"):
        parse_training_config(bad_temp)

    bad_profile = _training_body()
    bad_profile["structured_aux"] = {"teacher_public_heuristic_profiles": ["base", "unknown"]}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_heuristic_profiles contains unsupported profiles: unknown",
    ):
        parse_training_config(bad_profile)

    bad_profile_mode = _training_body()
    bad_profile_mode["structured_warmstart"] = {"teacher_public_heuristic_profile_mode": "round_robin"}
    with pytest.raises(
        ValueError,
        match="training.structured_warmstart.teacher_public_heuristic_profile_mode must be one of: cycle, mixture",
    ):
        parse_training_config(bad_profile_mode)

    bad_final = _training_body()
    bad_final["structured_aux"] = {"teacher_public_heuristic_final_coef": -0.01}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_heuristic_final_coef must be >= 0.0",
    ):
        parse_training_config(bad_final)

    bad_margin_coef = _training_body()
    bad_margin_coef["structured_aux"] = {"teacher_action_margin_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_action_margin_coef must be >= 0.0",
    ):
        parse_training_config(bad_margin_coef)

    bad_margin = _training_body()
    bad_margin["structured_aux"] = {"teacher_action_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_action_margin must be >= 0.0",
    ):
        parse_training_config(bad_margin)

    bad_same_family_margin_coef = _training_body()
    bad_same_family_margin_coef["structured_aux"] = {"teacher_same_family_action_margin_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_same_family_action_margin_coef must be >= 0.0",
    ):
        parse_training_config(bad_same_family_margin_coef)

    bad_same_family_margin = _training_body()
    bad_same_family_margin["structured_aux"] = {"teacher_same_family_action_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_same_family_action_margin must be >= 0.0",
    ):
        parse_training_config(bad_same_family_margin)

    bad_nonpass_coef = _training_body()
    bad_nonpass_coef["structured_aux"] = {"teacher_public_nonpass_over_pass_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_nonpass_over_pass_coef must be >= 0.0",
    ):
        parse_training_config(bad_nonpass_coef)

    bad_nonpass_margin = _training_body()
    bad_nonpass_margin["structured_aux"] = {"teacher_public_nonpass_over_pass_margin": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.teacher_public_nonpass_over_pass_margin must be >= 0.0",
    ):
        parse_training_config(bad_nonpass_margin)

    bad_anchor_coef = _training_body()
    bad_anchor_coef["structured_aux"] = {"policy_anchor_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_coef must be >= 0.0",
    ):
        parse_training_config(bad_anchor_coef)

    bad_anchor_top_action_coef = _training_body()
    bad_anchor_top_action_coef["structured_aux"] = {"policy_anchor_top_action_coef": -0.1}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_top_action_coef must be >= 0.0",
    ):
        parse_training_config(bad_anchor_top_action_coef)

    bad_anchor_temperature = _training_body()
    bad_anchor_temperature["structured_aux"] = {"policy_anchor_temperature": 0.0}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.policy_anchor_temperature must be > 0",
    ):
        parse_training_config(bad_anchor_temperature)

    bad_retention_coef = _training_body()
    bad_retention_coef["structured_aux"] = {"trajectory_retention_coef": -0.01}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.trajectory_retention_coef must be >= 0.0",
    ):
        parse_training_config(bad_retention_coef)

    bad_retention_source = _training_body()
    bad_retention_source["structured_aux"] = {"trajectory_retention_sources": ["champions", "unknown"]}
    with pytest.raises(
        ValueError,
        match="training.structured_aux.trajectory_retention_sources contains unsupported sources: unknown",
    ):
        parse_training_config(bad_retention_source)

    bad_focus_fraction = _training_body()
    bad_focus_fraction["structured_aux"] = {"trajectory_bc_focus_fraction": 1.25}
    with pytest.raises(ValueError, match="trajectory_bc_focus_fraction must be between"):
        parse_training_config(bad_focus_fraction)

    bad_focus_contract = _training_body()
    bad_focus_contract["structured_aux"] = {
        "trajectory_bc_focus_source_labels": ["repair_a"],
        "trajectory_bc_focus_fraction": 0.25,
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_b"], "fraction": 0.25},
        ],
    }
    with pytest.raises(ValueError, match="trajectory_bc_focus_groups cannot be combined"):
        parse_training_config(bad_focus_contract)

    bad_paired_swing_scope = _training_body()
    bad_paired_swing_scope["structured_aux"] = {"paired_swing_loss_scope": "trajectory"}
    with pytest.raises(ValueError, match="paired_swing_loss_scope"):
        parse_training_config(bad_paired_swing_scope)

    bad_focus_overlap = _training_body()
    bad_focus_overlap["structured_aux"] = {
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_a"], "fraction": 0.25},
            {"name": "fixed", "source_labels": ["repair_a"], "fraction": 0.25},
        ],
    }
    with pytest.raises(ValueError, match="contains labels in multiple groups: repair_a"):
        parse_training_config(bad_focus_overlap)

    bad_focus_sum = _training_body()
    bad_focus_sum["structured_aux"] = {
        "trajectory_bc_focus_groups": [
            {"name": "learned", "source_labels": ["repair_a"], "fraction": 0.70},
            {"name": "fixed", "source_labels": ["repair_b"], "fraction": 0.40},
        ],
    }
    with pytest.raises(ValueError, match="fractions must sum to <= 1.0"):
        parse_training_config(bad_focus_sum)


def test_parse_training_config_preserves_nested_unknown_and_minimum_errors() -> None:
    unknown = _training_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="training has unsupported keys: extra"):
        parse_training_config(unknown)

    bad_rollout = _training_body()
    rollout = _copy_section(bad_rollout, "rollout")
    rollout["extra"] = True
    bad_rollout["rollout"] = rollout
    with pytest.raises(ValueError, match="training.rollout has unsupported keys: extra"):
        parse_training_config(bad_rollout)

    bad_unroll = _training_body()
    rollout = _copy_section(bad_unroll, "rollout")
    rollout["unroll_length"] = 0
    bad_unroll["rollout"] = rollout
    with pytest.raises(ValueError, match="training.rollout.unroll_length must be >= 1, got 0"):
        parse_training_config(bad_unroll)
