"""Environment and reward stack config section parsers."""

from __future__ import annotations

from typing import Any

from .models import (
    DeckSetSizeConfig,
    EnvironmentConfig,
    RewardDiscountConfig,
    RewardsConfig,
    RewardShapingConfig,
    RewardTruncationConfig,
)
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_str_list,
    require_text,
)


def parse_environment_config(body: dict[str, Any]) -> EnvironmentConfig:
    reject_unknown_keys(
        body,
        allowed={
            "observation_visibility",
            "visibility",
            "truncate_on_max_steps",
            "max_raw_decisions_per_episode",
            "max_decisions",
            "max_decisions_per_episode",
            "max_learner_steps_per_episode",
            "max_ticks",
            "deck_set_size",
            "deck_pool",
            "opponent_deck_pool",
        },
        context="environment",
    )
    deck_set_size = require_mapping(body["deck_set_size"], context="environment.deck_set_size")
    reject_unknown_keys(deck_set_size, allowed={"bring_up", "paper"}, context="environment.deck_set_size")
    return EnvironmentConfig(
        observation_visibility=require_text(
            body["observation_visibility"],
            field_name="environment.observation_visibility",
        ),
        visibility=require_text(body["visibility"], field_name="environment.visibility"),
        truncate_on_max_steps=require_bool(
            body["truncate_on_max_steps"], field_name="environment.truncate_on_max_steps"
        ),
        max_raw_decisions_per_episode=require_int(
            body["max_raw_decisions_per_episode"],
            field_name="environment.max_raw_decisions_per_episode",
            minimum=1,
        ),
        max_decisions=require_int(body["max_decisions"], field_name="environment.max_decisions", minimum=1),
        max_decisions_per_episode=require_int(
            body["max_decisions_per_episode"],
            field_name="environment.max_decisions_per_episode",
            minimum=1,
        ),
        max_learner_steps_per_episode=require_int(
            body["max_learner_steps_per_episode"],
            field_name="environment.max_learner_steps_per_episode",
            minimum=1,
        ),
        max_ticks=require_int(body["max_ticks"], field_name="environment.max_ticks", minimum=1),
        deck_set_size=DeckSetSizeConfig(
            bring_up=require_int(deck_set_size["bring_up"], field_name="environment.deck_set_size.bring_up", minimum=1),
            paper=require_int(deck_set_size["paper"], field_name="environment.deck_set_size.paper", minimum=1),
        ),
        deck_pool=require_str_list(body.get("deck_pool", []), field_name="environment.deck_pool"),
        opponent_deck_pool=require_str_list(
            body.get("opponent_deck_pool", []),
            field_name="environment.opponent_deck_pool",
        ),
    )


def parse_rewards_config(body: dict[str, Any]) -> RewardsConfig:
    reject_unknown_keys(body, allowed={"objective", "discount", "shaping", "truncation"}, context="rewards")
    discount = require_mapping(body["discount"], context="rewards.discount")
    shaping = require_mapping(body["shaping"], context="rewards.shaping")
    truncation = require_mapping(body["truncation"], context="rewards.truncation")
    reject_unknown_keys(discount, allowed={"gamma"}, context="rewards.discount")
    reject_unknown_keys(
        shaping,
        allowed={
            "enable_damage_shaping",
            "damage_reward",
            "level_reward",
            "board_reward",
            "no_progress_penalty",
            "pass_with_nonpass_penalty",
            "mulligan_select_with_confirm_penalty",
            "terminal_outcome_backfill_reward",
            "terminal_outcome_trace_backfill_reward",
        },
        context="rewards.shaping",
    )
    reject_unknown_keys(
        truncation,
        allowed={"reward", "bootstrap_value", "bootstrap_rule"},
        context="rewards.truncation",
    )
    pass_with_nonpass_penalty = require_float(
        shaping.get("pass_with_nonpass_penalty", 0.0),
        field_name="rewards.shaping.pass_with_nonpass_penalty",
    )
    if pass_with_nonpass_penalty < 0.0:
        raise ValueError(f"rewards.shaping.pass_with_nonpass_penalty must be >= 0.0, got {pass_with_nonpass_penalty}")
    mulligan_select_with_confirm_penalty = require_float(
        shaping.get("mulligan_select_with_confirm_penalty", 0.0),
        field_name="rewards.shaping.mulligan_select_with_confirm_penalty",
    )
    if mulligan_select_with_confirm_penalty < 0.0:
        raise ValueError(
            "rewards.shaping.mulligan_select_with_confirm_penalty must be >= 0.0, "
            f"got {mulligan_select_with_confirm_penalty}"
        )
    terminal_outcome_backfill_reward = require_float(
        shaping.get("terminal_outcome_backfill_reward", 0.0),
        field_name="rewards.shaping.terminal_outcome_backfill_reward",
    )
    if terminal_outcome_backfill_reward < 0.0:
        raise ValueError(
            f"rewards.shaping.terminal_outcome_backfill_reward must be >= 0.0, got {terminal_outcome_backfill_reward}"
        )
    terminal_outcome_trace_backfill_reward = require_float(
        shaping.get("terminal_outcome_trace_backfill_reward", 0.0),
        field_name="rewards.shaping.terminal_outcome_trace_backfill_reward",
    )
    if terminal_outcome_trace_backfill_reward < 0.0:
        raise ValueError(
            "rewards.shaping.terminal_outcome_trace_backfill_reward must be >= 0.0, "
            f"got {terminal_outcome_trace_backfill_reward}"
        )
    return RewardsConfig(
        objective=require_text(body["objective"], field_name="rewards.objective"),
        discount=RewardDiscountConfig(
            gamma=require_float(discount["gamma"], field_name="rewards.discount.gamma"),
        ),
        shaping=RewardShapingConfig(
            enable_damage_shaping=require_bool(
                shaping["enable_damage_shaping"],
                field_name="rewards.shaping.enable_damage_shaping",
            ),
            damage_reward=require_float(shaping["damage_reward"], field_name="rewards.shaping.damage_reward"),
            level_reward=require_float(shaping.get("level_reward", 0.0), field_name="rewards.shaping.level_reward"),
            board_reward=require_float(shaping.get("board_reward", 0.0), field_name="rewards.shaping.board_reward"),
            no_progress_penalty=require_float(
                shaping.get("no_progress_penalty", 0.0),
                field_name="rewards.shaping.no_progress_penalty",
            ),
            pass_with_nonpass_penalty=pass_with_nonpass_penalty,
            mulligan_select_with_confirm_penalty=mulligan_select_with_confirm_penalty,
            terminal_outcome_backfill_reward=terminal_outcome_backfill_reward,
            terminal_outcome_trace_backfill_reward=terminal_outcome_trace_backfill_reward,
        ),
        truncation=RewardTruncationConfig(
            reward=require_float(truncation["reward"], field_name="rewards.truncation.reward"),
            bootstrap_value=require_bool(
                truncation["bootstrap_value"],
                field_name="rewards.truncation.bootstrap_value",
            ),
            bootstrap_rule=require_text(
                truncation["bootstrap_rule"],
                field_name="rewards.truncation.bootstrap_rule",
            ),
        ),
    )
