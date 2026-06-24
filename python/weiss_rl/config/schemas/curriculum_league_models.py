"""Curriculum and league config records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CurriculumStallMonitorConfig:
    enabled: bool
    truncation_rate_threshold: float
    consecutive_evals: int


@dataclass(frozen=True, slots=True)
class CurriculumCheckpointGuardConfig:
    enabled: bool
    rollback_score_margin: float
    rollback_truncation_rate_threshold: float
    rollback_max_prob_lt_half: float
    min_best_score: float
    promote_min_prob_gt_half: float
    promote_max_ci_half_width: float
    cooldown_updates: int
    stop_after_rollback: bool


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    simulator: dict[str, Any] = field(default_factory=dict)
    stall_monitor: CurriculumStallMonitorConfig = field(
        default_factory=lambda: CurriculumStallMonitorConfig(
            enabled=False,
            truncation_rate_threshold=1.0,
            consecutive_evals=2,
        )
    )
    checkpoint_guard: CurriculumCheckpointGuardConfig = field(
        default_factory=lambda: CurriculumCheckpointGuardConfig(
            enabled=False,
            rollback_score_margin=1.0,
            rollback_truncation_rate_threshold=1.0,
            rollback_max_prob_lt_half=1.0,
            min_best_score=1.0,
            promote_min_prob_gt_half=0.0,
            promote_max_ci_half_width=1.0,
            cooldown_updates=0,
            stop_after_rollback=False,
        )
    )


@dataclass(frozen=True, slots=True)
class LeagueWarmupConfig:
    first_updates: int
    initial_window_episodes: int
    ramp_target_updates: int
    ramp_target_window_episodes: int


@dataclass(frozen=True, slots=True)
class LeaguePoolConfig:
    recent_size: int
    champion_size: int
    champion_max_age_updates: int
    seed_snapshot_champion_import: str = "source_champions"
    seed_snapshot_import_filter: str = "all"
    seed_snapshot_registry_json: str = ""


@dataclass(frozen=True, slots=True)
class PromotionAnchorSetConfig:
    required: tuple[str, ...]
    optional_if_available: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionGateGuardrailsConfig:
    max_prob_anchor_loss_below_0_45: float
    max_truncation_rate: float


@dataclass(frozen=True, slots=True)
class PromotionGateConfig:
    uncertainty_method: str
    weighting: str
    seat_swap: bool
    folding: str
    guardrails: PromotionGateGuardrailsConfig
    record_file: str


@dataclass(frozen=True, slots=True)
class LeagueSamplingConfig:
    opponent_sampling: str
    pfsp_power: float
    pfsp_epsilon_uniform: float
    pfsp_stats_source: str
    pfsp_window_episodes: int
    mirror_mix_fraction: float
    mirror_mix_end_updates: int
    mirror_final_mix_fraction: float
    heuristic_public_start_updates: int
    heuristic_public_mix_fraction: float
    heuristic_public_mix_end_updates: int
    heuristic_public_final_mix_fraction: float
    heuristic_public_variant_mix_fraction: float
    heuristic_public_variant_mix_end_updates: int
    heuristic_public_variant_final_mix_fraction: float
    noleague_baseline_mix_fraction: float
    noleague_baseline_mix_end_updates: int
    warmup_snapshot_mix_fraction: float
    heuristic_public_reserved_envs_per_actor: int
    noleague_baseline_reserved_envs_per_actor: int
    champion_mix_fraction: float
    hard_negative_mix_fraction: float
    hard_negative_min_samples: int
    hard_negative_max_win_rate: float
    hard_negative_focus_policy_ids: tuple[str, ...]
    hard_negative_focus_weight_multiplier: float
    row_deficit_policy_weights: tuple[tuple[str, float], ...]
    hard_negative_overlaps_champions: bool


@dataclass(frozen=True, slots=True)
class LeaguePromotionConfig:
    enabled: bool
    paired_seeds: int
    threshold: str
    anchor_set_v1: PromotionAnchorSetConfig
    seed_file: str
    gate: PromotionGateConfig


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    enabled: bool
    pool: LeaguePoolConfig
    sampling: LeagueSamplingConfig
    warmup: LeagueWarmupConfig
    promotion: LeaguePromotionConfig

    @property
    def snapshot_pool_recent_size(self) -> int:
        return int(self.pool.recent_size)

    @property
    def snapshot_pool_champion_size(self) -> int:
        return int(self.pool.champion_size)

    @property
    def opponent_sampling(self) -> str:
        return self.sampling.opponent_sampling

    @property
    def pfsp_power(self) -> float:
        return float(self.sampling.pfsp_power)

    @property
    def pfsp_epsilon_uniform(self) -> float:
        return float(self.sampling.pfsp_epsilon_uniform)

    @property
    def pfsp_stats_source(self) -> str:
        return self.sampling.pfsp_stats_source

    @property
    def pfsp_window_episodes(self) -> int:
        return int(self.sampling.pfsp_window_episodes)

    @property
    def promotion_gate_enabled(self) -> bool:
        return bool(self.promotion.enabled)

    @property
    def promotion_gate_paired_seeds(self) -> int:
        return int(self.promotion.paired_seeds)

    @property
    def promotion_threshold(self) -> str:
        return self.promotion.threshold

    @property
    def promotion_anchor_set_v1(self) -> PromotionAnchorSetConfig:
        return self.promotion.anchor_set_v1

    @property
    def promotion_seed_file(self) -> str:
        return self.promotion.seed_file

    @property
    def promotion_gate(self) -> PromotionGateConfig:
        return self.promotion.gate


__all__ = [
    "CurriculumCheckpointGuardConfig",
    "CurriculumConfig",
    "CurriculumStallMonitorConfig",
    "LeagueConfig",
    "LeaguePoolConfig",
    "LeaguePromotionConfig",
    "LeagueSamplingConfig",
    "LeagueWarmupConfig",
    "PromotionAnchorSetConfig",
    "PromotionGateConfig",
    "PromotionGateGuardrailsConfig",
]
