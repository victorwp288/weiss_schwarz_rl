"""Build simulator environment config dictionaries from stack config."""

from __future__ import annotations

import json
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.envs.simulator_reward_contract import reward_payload_json_from_stack


def build_env_config_from_stack(
    stack: StackConfig,
    *,
    seed: int,
    actor_id: int | None = None,
    deck: str | None = None,
    opponent_deck: str | None = None,
) -> dict[str, Any]:
    environment_config = stack.config.environment
    if environment_config is None:
        raise RuntimeError("stack config is missing environment")
    env_config: dict[str, Any] = {
        "max_decisions": int(environment_config.max_decisions),
        "max_ticks": int(environment_config.max_ticks),
        "observation_visibility": environment_config.observation_visibility,
        "seed": int(seed),
    }
    reward_json = reward_payload_json_from_stack(stack)
    curriculum_json = _curriculum_payload_from_stack(stack)
    if reward_json is not None:
        env_config["reward_json"] = reward_json
    if curriculum_json is not None:
        env_config["curriculum_json"] = curriculum_json
    resolved_deck = deck or _cycle_deck_choice(environment_config.deck_pool, actor_id=actor_id)
    resolved_opponent_deck = opponent_deck or _cycle_deck_choice(
        environment_config.opponent_deck_pool,
        actor_id=actor_id,
    )
    if resolved_deck is not None:
        env_config["deck"] = str(resolved_deck)
    if resolved_opponent_deck is not None:
        env_config["opponent_deck"] = str(resolved_opponent_deck)
    return env_config


def _curriculum_payload_from_stack(stack: StackConfig) -> str | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.simulator:
        return None
    return json.dumps(curriculum.simulator, sort_keys=True)


def _cycle_deck_choice(deck_pool: tuple[str, ...], *, actor_id: int | None) -> str | None:
    if not deck_pool:
        return None
    if actor_id is None:
        return str(deck_pool[0])
    return str(deck_pool[int(actor_id) % len(deck_pool)])


__all__ = ["build_env_config_from_stack"]
