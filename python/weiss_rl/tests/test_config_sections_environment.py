from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from weiss_rl.config.sections_environment import parse_environment_config, parse_rewards_config


def _copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def _environment_body() -> dict[str, object]:
    return {
        "observation_visibility": "public",
        "visibility": "public",
        "truncate_on_max_steps": True,
        "max_raw_decisions_per_episode": 128,
        "max_decisions": 64,
        "max_decisions_per_episode": 64,
        "max_learner_steps_per_episode": 32,
        "max_ticks": 1024,
        "deck_set_size": {"bring_up": 8, "paper": 50},
    }


def _rewards_body() -> dict[str, object]:
    return {
        "objective": "terminal_pm1",
        "discount": {"gamma": 0.99},
        "shaping": {
            "enable_damage_shaping": True,
            "damage_reward": 0.01,
        },
        "truncation": {
            "reward": -0.1,
            "bootstrap_value": True,
            "bootstrap_rule": "bootstrap",
        },
    }


def test_parse_environment_config_applies_deck_pool_defaults() -> None:
    config = parse_environment_config(_environment_body())

    assert config.observation_visibility == "public"
    assert config.truncate_on_max_steps is True
    assert config.deck_set_size.bring_up == 8
    assert config.deck_set_size.paper == 50
    assert config.deck_pool == ()
    assert config.opponent_deck_pool == ()


def test_parse_environment_config_accepts_deck_pools() -> None:
    body = _environment_body()
    body["deck_pool"] = ["set_a", " set_b "]
    body["opponent_deck_pool"] = ["opp"]

    config = parse_environment_config(body)

    assert config.deck_pool == ("set_a", "set_b")
    assert config.opponent_deck_pool == ("opp",)


def test_parse_environment_config_preserves_validation_errors() -> None:
    unknown = _environment_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="environment has unsupported keys: extra"):
        parse_environment_config(unknown)

    bad_deck = _environment_body()
    bad_deck["deck_set_size"] = {"bring_up": 8, "paper": 50, "extra": 1}
    with pytest.raises(ValueError, match="environment.deck_set_size has unsupported keys: extra"):
        parse_environment_config(bad_deck)

    bad_max = _environment_body()
    bad_max["max_ticks"] = 0
    with pytest.raises(ValueError, match="environment.max_ticks must be >= 1, got 0"):
        parse_environment_config(bad_max)


def test_parse_rewards_config_applies_shaping_defaults() -> None:
    config = parse_rewards_config(_rewards_body())

    assert config.objective == "terminal_pm1"
    assert config.discount.gamma == pytest.approx(0.99)
    assert config.shaping.enable_damage_shaping is True
    assert config.shaping.damage_reward == pytest.approx(0.01)
    assert config.shaping.level_reward == pytest.approx(0.0)
    assert config.shaping.board_reward == pytest.approx(0.0)
    assert config.shaping.no_progress_penalty == pytest.approx(0.0)
    assert config.shaping.pass_with_nonpass_penalty == pytest.approx(0.0)
    assert config.shaping.mulligan_select_with_confirm_penalty == pytest.approx(0.0)
    assert config.shaping.terminal_outcome_backfill_reward == pytest.approx(0.0)
    assert config.shaping.terminal_outcome_trace_backfill_reward == pytest.approx(0.0)
    assert config.truncation.reward == pytest.approx(-0.1)
    assert config.truncation.bootstrap_value is True
    assert config.truncation.bootstrap_rule == "bootstrap"


def test_parse_rewards_config_accepts_optional_shaping_values() -> None:
    body = _rewards_body()
    shaping = _copy_section(body, "shaping")
    shaping.update(
        {
            "level_reward": 0.02,
            "board_reward": 0.03,
            "no_progress_penalty": 0.04,
            "pass_with_nonpass_penalty": 0.05,
            "mulligan_select_with_confirm_penalty": 0.06,
            "terminal_outcome_backfill_reward": 0.07,
            "terminal_outcome_trace_backfill_reward": 0.08,
        }
    )
    body["shaping"] = shaping

    config = parse_rewards_config(body)

    assert config.shaping.level_reward == pytest.approx(0.02)
    assert config.shaping.board_reward == pytest.approx(0.03)
    assert config.shaping.no_progress_penalty == pytest.approx(0.04)
    assert config.shaping.pass_with_nonpass_penalty == pytest.approx(0.05)
    assert config.shaping.mulligan_select_with_confirm_penalty == pytest.approx(0.06)
    assert config.shaping.terminal_outcome_backfill_reward == pytest.approx(0.07)
    assert config.shaping.terminal_outcome_trace_backfill_reward == pytest.approx(0.08)


def test_parse_rewards_config_preserves_validation_errors() -> None:
    unknown = _rewards_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="rewards has unsupported keys: extra"):
        parse_rewards_config(unknown)

    bad_discount = _rewards_body()
    bad_discount["discount"] = {"gamma": 0.99, "extra": 1}
    with pytest.raises(ValueError, match="rewards.discount has unsupported keys: extra"):
        parse_rewards_config(bad_discount)

    bad_flag = _rewards_body()
    bad_flag["truncation"] = {"reward": -0.1, "bootstrap_value": "true", "bootstrap_rule": "bootstrap"}
    with pytest.raises(ValueError, match="rewards.truncation.bootstrap_value must be a boolean, got str"):
        parse_rewards_config(bad_flag)
