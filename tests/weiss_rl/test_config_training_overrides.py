from __future__ import annotations

import pytest
from weiss_rl.config.sections_training import parse_training_config

from tests.weiss_rl.config_training_test_support import copy_section, training_body


def test_parse_training_config_accepts_nested_overrides() -> None:
    body = training_body()
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
    precision = copy_section(body, "precision")
    precision["compile_actor_inference"] = True
    body["precision"] = precision
    exploration = copy_section(body, "exploration")
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
