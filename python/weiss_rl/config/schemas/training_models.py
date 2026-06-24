"""Training config records and compatibility accessors."""

from __future__ import annotations

from dataclasses import dataclass, field

from weiss_rl.config.schemas.training_aux_models import (
    TrainingStructuredAuxConfig,
    TrainingStructuredWarmstartConfig,
    TrainingTrajectoryBcFocusGroupConfig,
)
from weiss_rl.config.schemas.training_compat_accessors import TrainingConfigCompatibilityMixin


@dataclass(frozen=True, slots=True)
class TrainingRolloutConfig:
    unroll_length: int
    batch_unrolls_per_update: int


@dataclass(frozen=True, slots=True)
class TrainingOptimizerConfig:
    name: str
    learning_rate: float
    grad_norm_clip: float
    value_loss_coef: float


@dataclass(frozen=True, slots=True)
class TrainingExplorationConfig:
    entropy_coef: float
    entropy_anneal_to: float
    entropy_anneal_steps_updates: int
    entropy_scope: str = "candidate"
    actor_sampling_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingPrecisionConfig:
    mixed_precision: bool
    compile_learner: bool
    compile_actor_inference: bool
    masking_math_float32: bool


@dataclass(frozen=True, slots=True)
class TrainingStructuredMetricsConfig:
    mode: str = "off"


@dataclass(frozen=True, slots=True)
class TrainingTeacherAuxConfig:
    mode: str = "always"


@dataclass(frozen=True, slots=True)
class TrainingActionSurfaceConfig:
    mulligan_force_confirm_after_select: bool = False
    force_pass_over_main_move_only: bool = False
    main_move_only_max_consecutive: int = 0
    force_attack_over_pass_when_attack_legal: bool = False


@dataclass(frozen=True, slots=True)
class TrainingCheckpointingConfig:
    checkpoint_interval_updates: int
    snapshot_interval_updates: int
    actor_reload_interval_updates: int


@dataclass(frozen=True, slots=True)
class TrainingVTraceConfig:
    rho_bar: float
    c_bar: float


@dataclass(frozen=True, slots=True)
class TrainingPpoConfig:
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    epochs: int = 4
    target_kl: float = 0.0
    normalize_advantages: bool = True


@dataclass(frozen=True, slots=True)
class TrainingConfig(TrainingConfigCompatibilityMixin):
    algorithm: str
    rollout: TrainingRolloutConfig
    optimizer: TrainingOptimizerConfig
    exploration: TrainingExplorationConfig
    precision: TrainingPrecisionConfig
    checkpointing: TrainingCheckpointingConfig
    vtrace: TrainingVTraceConfig
    ppo: TrainingPpoConfig
    structured_aux: TrainingStructuredAuxConfig
    structured_warmstart: TrainingStructuredWarmstartConfig
    structured_metrics: TrainingStructuredMetricsConfig = field(default_factory=TrainingStructuredMetricsConfig)
    teacher_aux: TrainingTeacherAuxConfig = field(default_factory=TrainingTeacherAuxConfig)
    action_surface: TrainingActionSurfaceConfig = field(default_factory=TrainingActionSurfaceConfig)
    fixed_opponent_backend: str = "python_scalar"
    fixed_model_opponent_action_selection: str = "sample"
    actor_policy_backend: str = "model"
    actor_heuristic_fraction: float = 1.0
    actor_heuristic_start_updates: int = 0
    actor_heuristic_end_updates: int = -1
    actor_heuristic_final_fraction: float = 1.0
    train_on_heuristic_actor_rows: bool = True
    diverse_opponent_actor_count: int = 0
    diverse_model_actor_count: int = 0
    diverse_opponent_batch_fraction: float = 0.0
    diverse_opponent_batch_wait_ms: int = 0
    heuristic_actor_hidden_state_tracking: bool = True
    profile_timers: bool = False
    torch_profiler: bool = False


__all__ = [
    "TrainingActionSurfaceConfig",
    "TrainingCheckpointingConfig",
    "TrainingConfig",
    "TrainingExplorationConfig",
    "TrainingOptimizerConfig",
    "TrainingPpoConfig",
    "TrainingPrecisionConfig",
    "TrainingRolloutConfig",
    "TrainingStructuredAuxConfig",
    "TrainingStructuredMetricsConfig",
    "TrainingStructuredWarmstartConfig",
    "TrainingTeacherAuxConfig",
    "TrainingTrajectoryBcFocusGroupConfig",
    "TrainingVTraceConfig",
]
