"""Environment and reward config records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DeckSetSizeConfig:
    bring_up: int
    paper: int


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    observation_visibility: str
    visibility: str
    truncate_on_max_steps: bool
    max_raw_decisions_per_episode: int
    max_decisions: int
    max_decisions_per_episode: int
    max_learner_steps_per_episode: int
    max_ticks: int
    deck_set_size: DeckSetSizeConfig
    deck_pool: tuple[str, ...] = field(default_factory=tuple)
    opponent_deck_pool: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RewardDiscountConfig:
    gamma: float


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    enable_damage_shaping: bool
    damage_reward: float
    level_reward: float
    board_reward: float
    no_progress_penalty: float
    pass_with_nonpass_penalty: float = 0.0
    mulligan_select_with_confirm_penalty: float = 0.0
    terminal_outcome_backfill_reward: float = 0.0
    terminal_outcome_trace_backfill_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardTruncationConfig:
    reward: float
    bootstrap_value: bool
    bootstrap_rule: str


@dataclass(frozen=True, slots=True)
class RewardsConfig:
    objective: str
    discount: RewardDiscountConfig
    shaping: RewardShapingConfig
    truncation: RewardTruncationConfig

    @property
    def gamma(self) -> float:
        return float(self.discount.gamma)


__all__ = [
    "DeckSetSizeConfig",
    "EnvironmentConfig",
    "RewardDiscountConfig",
    "RewardShapingConfig",
    "RewardTruncationConfig",
    "RewardsConfig",
]
