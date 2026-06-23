from __future__ import annotations

import pytest
from weiss_rl.config.sections_training import parse_training_config

from tests.weiss_rl.config_training_test_support import training_body


def test_parse_training_config_applies_existing_defaults() -> None:
    config = parse_training_config(training_body())

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
