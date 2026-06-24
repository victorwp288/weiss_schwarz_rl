"""Translate training reward config into the simulator reward contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from weiss_rl.config import StackConfig
from weiss_rl.config.schemas.environment_models import RewardsConfig

SUPPORTED_SIMULATOR_REWARD_OBJECTIVES = frozenset({"terminal_pm1", "terminal_only_pm1"})


@dataclass(frozen=True, slots=True)
class SimulatorRewardPayload:
    terminal_win: float
    terminal_loss: float
    terminal_draw: float
    terminal_timeout: float
    enable_shaping: bool
    damage_reward: float
    level_reward: float
    board_reward: float
    no_progress_penalty: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def simulator_reward_payload_from_rewards(rewards: RewardsConfig) -> SimulatorRewardPayload:
    """Build the exact reward payload accepted by `weiss_sim`."""

    objective = str(rewards.objective).strip().lower()
    if objective not in SUPPORTED_SIMULATOR_REWARD_OBJECTIVES:
        raise ValueError(f"Unsupported rewards.objective {rewards.objective!r}")

    shaping_enabled = bool(rewards.shaping.enable_damage_shaping)
    damage_reward = float(rewards.shaping.damage_reward)
    level_reward = float(rewards.shaping.level_reward)
    board_reward = float(rewards.shaping.board_reward)
    no_progress_penalty = float(rewards.shaping.no_progress_penalty)

    if objective == "terminal_only_pm1":
        shaping_enabled = False
        damage_reward = 0.0
        level_reward = 0.0
        board_reward = 0.0
        no_progress_penalty = 0.0

    return SimulatorRewardPayload(
        terminal_win=1.0,
        terminal_loss=-1.0,
        terminal_draw=0.0,
        terminal_timeout=float(rewards.truncation.reward),
        enable_shaping=shaping_enabled,
        damage_reward=damage_reward,
        level_reward=level_reward,
        board_reward=board_reward,
        no_progress_penalty=no_progress_penalty,
    )


def reward_payload_json_from_stack(stack: StackConfig) -> str | None:
    rewards = stack.config.rewards
    if rewards is None:
        return None
    return simulator_reward_payload_from_rewards(rewards).to_json()


__all__ = [
    "SUPPORTED_SIMULATOR_REWARD_OBJECTIVES",
    "SimulatorRewardPayload",
    "reward_payload_json_from_stack",
    "simulator_reward_payload_from_rewards",
]
