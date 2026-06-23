from __future__ import annotations

import pytest
from weiss_rl.config.sections_curriculum import normalize_curriculum_payload, parse_curriculum_config


def test_normalize_curriculum_payload_preserves_supported_nested_values() -> None:
    payload = {
        "max_no_progress": 12,
        "enabled": True,
        "profile": ["a", 2, 0.5, None, {"inner": "value"}],
    }

    assert normalize_curriculum_payload(payload, field_name="curriculum.simulator") == payload


def test_normalize_curriculum_payload_rejects_bad_keys_and_values() -> None:
    with pytest.raises(ValueError, match="curriculum.simulator.<key> must be a non-empty string"):
        normalize_curriculum_payload({1: "value"}, field_name="curriculum.simulator")

    with pytest.raises(ValueError, match="curriculum.simulator contains unsupported value type: set"):
        normalize_curriculum_payload({"bad"}, field_name="curriculum.simulator")


def test_parse_curriculum_config_defaults_when_absent() -> None:
    config = parse_curriculum_config(None)

    assert config.simulator == {}
    assert config.stall_monitor.enabled is False
    assert config.stall_monitor.truncation_rate_threshold == pytest.approx(1.0)
    assert config.stall_monitor.consecutive_evals == 2
    assert config.checkpoint_guard.enabled is False
    assert config.checkpoint_guard.cooldown_updates == 0
    assert config.checkpoint_guard.stop_after_rollback is False


def test_parse_curriculum_config_applies_nested_values() -> None:
    config = parse_curriculum_config(
        {
            "simulator": {"max_no_progress_decisions": 64, "nested": {"mode": "strict"}},
            "stall_monitor": {
                "enabled": True,
                "truncation_rate_threshold": 0.2,
                "consecutive_evals": 3,
            },
            "checkpoint_guard": {
                "enabled": True,
                "rollback_score_margin": 0.1,
                "rollback_truncation_rate_threshold": 0.3,
                "rollback_max_prob_lt_half": 0.4,
                "min_best_score": 0.55,
                "promote_min_prob_gt_half": 0.6,
                "promote_max_ci_half_width": 0.24,
                "cooldown_updates": 20,
                "stop_after_rollback": True,
            },
        }
    )

    assert config.simulator == {"max_no_progress_decisions": 64, "nested": {"mode": "strict"}}
    assert config.stall_monitor.enabled is True
    assert config.stall_monitor.consecutive_evals == 3
    assert config.checkpoint_guard.enabled is True
    assert config.checkpoint_guard.rollback_score_margin == pytest.approx(0.1)
    assert config.checkpoint_guard.cooldown_updates == 20
    assert config.checkpoint_guard.stop_after_rollback is True


def test_parse_curriculum_config_preserves_validation_errors() -> None:
    with pytest.raises(ValueError, match="curriculum has unsupported keys: extra"):
        parse_curriculum_config({"extra": True})

    with pytest.raises(ValueError, match="curriculum.stall_monitor has unsupported keys: extra"):
        parse_curriculum_config({"stall_monitor": {"extra": True}})

    with pytest.raises(ValueError, match="curriculum.checkpoint_guard has unsupported keys: extra"):
        parse_curriculum_config({"checkpoint_guard": {"extra": True}})

    with pytest.raises(ValueError, match="curriculum.stall_monitor.consecutive_evals must be >= 1, got 0"):
        parse_curriculum_config({"stall_monitor": {"consecutive_evals": 0}})

    with pytest.raises(ValueError, match="curriculum.checkpoint_guard.cooldown_updates must be >= 0, got -1"):
        parse_curriculum_config({"checkpoint_guard": {"cooldown_updates": -1}})
