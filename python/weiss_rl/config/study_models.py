"""Top-level study and stack config records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.config.common_models import ExperimentConfig, ModelConfig, SystemConfig
from weiss_rl.config.curriculum_league_models import CurriculumConfig, LeagueConfig
from weiss_rl.config.environment_models import EnvironmentConfig, RewardsConfig
from weiss_rl.config.evaluation_models import EvaluationConfig, ReproducibilityConfig
from weiss_rl.config.training_models import TrainingConfig


@dataclass(frozen=True, slots=True)
class MetagameNashConfig:
    impl: str
    backend: str
    threads: int
    value_tolerance: float
    tie_break: str


@dataclass(frozen=True, slots=True)
class MetagameAlphaRankConfig:
    impl: str
    m: int
    alpha: float
    local_selection: bool
    use_inf_alpha: bool
    inf_alpha_eps: float


@dataclass(frozen=True, slots=True)
class MetagameConfig:
    payoff_uncertainty_method: str
    sampling_m: int
    optional_secondary_uncertainty_method: str
    dirichlet_alpha_wldt: float
    primary_analysis: str
    secondary_analysis: str
    nash: MetagameNashConfig
    alpharank: MetagameAlphaRankConfig


@dataclass(frozen=True, slots=True)
class SensitivityCaseConfig:
    description: str
    draw_score: float
    truncation_score: float | None = None
    truncation_handling: str | None = None


@dataclass(frozen=True, slots=True)
class SensitivityReportConfig:
    required_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SensitivityConfig:
    cases: dict[str, SensitivityCaseConfig]
    report: SensitivityReportConfig


@dataclass(frozen=True, slots=True)
class StudyConfig:
    root: Path
    schema_version: int | None
    description: str
    metagame: MetagameConfig
    sensitivity: SensitivityConfig


@dataclass(frozen=True, slots=True)
class LockedConfig:
    experiment: ExperimentConfig | None = None
    system: SystemConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    environment: EnvironmentConfig | None = None
    rewards: RewardsConfig | None = None
    curriculum: CurriculumConfig | None = None
    league: LeagueConfig | None = None
    evaluation: EvaluationConfig | None = None
    reproducibility: ReproducibilityConfig | None = None


@dataclass(frozen=True, slots=True)
class StackConfig:
    root: Path
    schema_version: int | None
    description: str
    lock_intent: dict[str, Any]
    components: dict[str, Path]
    seed_sets: dict[str, Path]
    component_docs: dict[str, dict[str, Any]]
    config: LockedConfig


__all__ = [
    "LockedConfig",
    "MetagameAlphaRankConfig",
    "MetagameConfig",
    "MetagameNashConfig",
    "SensitivityCaseConfig",
    "SensitivityConfig",
    "SensitivityReportConfig",
    "StackConfig",
    "StudyConfig",
]
