"""Training config records and compatibility accessors."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class TrainingTrajectoryBcFocusGroupConfig:
    name: str
    source_labels: tuple[str, ...]
    fraction: float


@dataclass(frozen=True, slots=True)
class TrainingStructuredAuxConfig:
    enabled: bool = False
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_action_margin_coef: float = 0.0
    teacher_action_margin: float = 0.5
    teacher_same_family_action_margin_coef: float = 0.0
    teacher_same_family_action_margin: float = 0.5
    teacher_supervised_start_updates: int = 0
    teacher_supervised_end_updates: int = -1
    teacher_supervised_final_scale: float = 1.0
    teacher_exact_action_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_start_updates: int = 0
    teacher_public_heuristic_end_updates: int = -1
    teacher_public_heuristic_final_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_nonpass_over_pass_coef: float = 0.0
    teacher_public_nonpass_over_pass_margin: float = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_anchor_coef: float = 0.0
    policy_anchor_top_action_coef: float = 0.0
    policy_anchor_temperature: float = 1.0
    trajectory_retention_coef: float = 0.0
    trajectory_retention_policy_ids: tuple[str, ...] = field(default_factory=tuple)
    trajectory_retention_sources: tuple[str, ...] = field(default_factory=lambda: ("champions",))
    trajectory_bc_dataset_path: str = ""
    trajectory_bc_every_updates: int = 0
    trajectory_bc_aux_updates: int = 1
    trajectory_bc_batch_episodes: int = 8
    trajectory_bc_seed: int = 20260516
    trajectory_bc_focus_source_labels: tuple[str, ...] = field(default_factory=tuple)
    trajectory_bc_focus_fraction: float = 0.0
    trajectory_bc_focus_groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...] = field(default_factory=tuple)
    trajectory_bc_teacher_family_coef: float = 0.05
    trajectory_bc_teacher_slot_coef: float = 0.05
    trajectory_bc_teacher_move_source_coef: float = 0.02
    trajectory_bc_teacher_attack_type_coef: float = 0.02
    trajectory_bc_teacher_action_coef: float = 0.20
    trajectory_bc_teacher_same_family_action_coef: float = 0.60
    trajectory_bc_teacher_same_family_action_margin_coef: float = 0.10
    trajectory_bc_teacher_same_family_action_margin: float = 0.5
    paired_swing_dataset_path: str = ""
    paired_swing_every_updates: int = 0
    paired_swing_aux_updates: int = 1
    paired_swing_batch_episodes: int = 8
    paired_swing_seed: int = 20260519
    paired_swing_focus_source_labels: tuple[str, ...] = field(default_factory=tuple)
    paired_swing_focus_fraction: float = 0.0
    paired_swing_focus_groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...] = field(default_factory=tuple)
    paired_swing_margin: float = 0.35
    paired_swing_coef: float = 0.08
    paired_swing_positive_action_source: str = "teacher_action"
    paired_swing_negative_action_source: str = "actions"
    paired_swing_conflict_filter: str = "none"
    paired_swing_loss_scope: str = "row"
    paired_swing_compare_to: str = "negative"
    paired_outcome_preference_dataset_path: str = ""
    paired_outcome_preference_every_updates: int = 0
    paired_outcome_preference_aux_updates: int = 1
    paired_outcome_preference_batch_episodes: int = 8
    paired_outcome_preference_seed: int = 20260520
    paired_outcome_preference_coef: float = 0.05
    paired_outcome_preference_beta: float = 0.1
    paired_outcome_preference_aggregation: str = "mean"
    paired_outcome_preference_group_balance: bool = False


@dataclass(frozen=True, slots=True)
class TrainingStructuredWarmstartConfig:
    enabled: bool = False
    updates: int = 0
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
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

    @property
    def unroll_length(self) -> int:
        return int(self.rollout.unroll_length)

    @property
    def batch_unrolls_per_update(self) -> int:
        return int(self.rollout.batch_unrolls_per_update)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.learning_rate)

    @property
    def grad_norm_clip(self) -> float:
        return float(self.optimizer.grad_norm_clip)

    @property
    def value_loss_coef(self) -> float:
        return float(self.optimizer.value_loss_coef)

    @property
    def entropy_coef(self) -> float:
        return float(self.exploration.entropy_coef)

    @property
    def entropy_anneal_to(self) -> float:
        return float(self.exploration.entropy_anneal_to)

    @property
    def entropy_anneal_steps_updates(self) -> int:
        return int(self.exploration.entropy_anneal_steps_updates)

    @property
    def entropy_scope(self) -> str:
        return str(self.exploration.entropy_scope)

    @property
    def actor_sampling_temperature(self) -> float:
        return float(self.exploration.actor_sampling_temperature)

    @property
    def mixed_precision(self) -> bool:
        return bool(self.precision.mixed_precision)

    @property
    def compile_learner(self) -> bool:
        return bool(self.precision.compile_learner)

    @property
    def compile_actor_inference(self) -> bool:
        return bool(self.precision.compile_actor_inference)

    @property
    def masking_math_float32(self) -> bool:
        return bool(self.precision.masking_math_float32)

    @property
    def checkpoint_interval_updates(self) -> int:
        return int(self.checkpointing.checkpoint_interval_updates)

    @property
    def snapshot_interval_updates(self) -> int:
        return int(self.checkpointing.snapshot_interval_updates)

    @property
    def actor_reload_interval_updates(self) -> int:
        return int(self.checkpointing.actor_reload_interval_updates)

    @property
    def vtrace_rho_bar(self) -> float:
        return float(self.vtrace.rho_bar)

    @property
    def vtrace_c_bar(self) -> float:
        return float(self.vtrace.c_bar)

    @property
    def ppo_clip_epsilon(self) -> float:
        return float(self.ppo.clip_epsilon)

    @property
    def ppo_value_clip_epsilon(self) -> float:
        return float(self.ppo.value_clip_epsilon)

    @property
    def ppo_gae_lambda(self) -> float:
        return float(self.ppo.gae_lambda)

    @property
    def ppo_epochs(self) -> int:
        return int(self.ppo.epochs)

    @property
    def ppo_target_kl(self) -> float:
        return float(self.ppo.target_kl)

    @property
    def ppo_normalize_advantages(self) -> bool:
        return bool(self.ppo.normalize_advantages)

    @property
    def structured_aux_enabled(self) -> bool:
        return bool(self.structured_aux.enabled)

    @property
    def structured_metrics_mode(self) -> str:
        return self.structured_metrics.mode

    @property
    def teacher_aux_mode(self) -> str:
        return self.teacher_aux.mode

    @property
    def mulligan_force_confirm_after_select(self) -> bool:
        return bool(self.action_surface.mulligan_force_confirm_after_select)

    @property
    def force_pass_over_main_move_only(self) -> bool:
        return bool(self.action_surface.force_pass_over_main_move_only)

    @property
    def main_move_only_max_consecutive(self) -> int:
        return int(self.action_surface.main_move_only_max_consecutive)

    @property
    def force_attack_over_pass_when_attack_legal(self) -> bool:
        return bool(self.action_surface.force_attack_over_pass_when_attack_legal)

    @property
    def teacher_family_coef(self) -> float:
        return float(self.structured_aux.teacher_family_coef)

    @property
    def teacher_slot_coef(self) -> float:
        return float(self.structured_aux.teacher_slot_coef)

    @property
    def teacher_hand_coef(self) -> float:
        return float(self.structured_aux.teacher_hand_coef)

    @property
    def teacher_move_source_coef(self) -> float:
        return float(self.structured_aux.teacher_move_source_coef)

    @property
    def teacher_attack_type_coef(self) -> float:
        return float(self.structured_aux.teacher_attack_type_coef)

    @property
    def teacher_action_coef(self) -> float:
        return float(self.structured_aux.teacher_action_coef)

    @property
    def teacher_same_family_action_coef(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_coef)

    @property
    def teacher_action_margin_coef(self) -> float:
        return float(self.structured_aux.teacher_action_margin_coef)

    @property
    def teacher_action_margin(self) -> float:
        return float(self.structured_aux.teacher_action_margin)

    @property
    def teacher_same_family_action_margin_coef(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_margin_coef)

    @property
    def teacher_same_family_action_margin(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_margin)

    @property
    def teacher_supervised_start_updates(self) -> int:
        return int(self.structured_aux.teacher_supervised_start_updates)

    @property
    def teacher_supervised_end_updates(self) -> int:
        return int(self.structured_aux.teacher_supervised_end_updates)

    @property
    def teacher_supervised_final_scale(self) -> float:
        return float(self.structured_aux.teacher_supervised_final_scale)

    @property
    def teacher_exact_action_families(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_exact_action_families)

    @property
    def teacher_public_heuristic_coef(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_coef)

    @property
    def teacher_public_heuristic_start_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_start_updates)

    @property
    def teacher_public_heuristic_end_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_end_updates)

    @property
    def teacher_public_heuristic_final_coef(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_final_coef)

    @property
    def teacher_public_heuristic_temperature(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_temperature)

    @property
    def teacher_public_nonpass_over_pass_coef(self) -> float:
        return float(self.structured_aux.teacher_public_nonpass_over_pass_coef)

    @property
    def teacher_public_nonpass_over_pass_margin(self) -> float:
        return float(self.structured_aux.teacher_public_nonpass_over_pass_margin)

    @property
    def teacher_public_heuristic_families(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_public_heuristic_families)

    @property
    def teacher_public_heuristic_profiles(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_public_heuristic_profiles)

    @property
    def teacher_public_heuristic_profile_mode(self) -> str:
        return str(self.structured_aux.teacher_public_heuristic_profile_mode)

    @property
    def teacher_public_heuristic_profiles_end_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_profiles_end_updates)

    @property
    def policy_anchor_coef(self) -> float:
        return float(self.structured_aux.policy_anchor_coef)

    @property
    def policy_anchor_top_action_coef(self) -> float:
        return float(self.structured_aux.policy_anchor_top_action_coef)

    @property
    def policy_anchor_temperature(self) -> float:
        return float(self.structured_aux.policy_anchor_temperature)

    @property
    def trajectory_retention_coef(self) -> float:
        return float(self.structured_aux.trajectory_retention_coef)

    @property
    def trajectory_retention_policy_ids(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.trajectory_retention_policy_ids)

    @property
    def trajectory_retention_sources(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.trajectory_retention_sources)

    @property
    def trajectory_bc_dataset_path(self) -> str:
        return str(self.structured_aux.trajectory_bc_dataset_path)

    @property
    def trajectory_bc_enabled(self) -> bool:
        return bool(str(self.structured_aux.trajectory_bc_dataset_path).strip()) and (
            int(self.structured_aux.trajectory_bc_every_updates) > 0
        )

    @property
    def paired_swing_enabled(self) -> bool:
        return bool(str(self.structured_aux.paired_swing_dataset_path).strip()) and (
            int(self.structured_aux.paired_swing_every_updates) > 0
        )

    @property
    def paired_outcome_preference_enabled(self) -> bool:
        return bool(str(self.structured_aux.paired_outcome_preference_dataset_path).strip()) and (
            int(self.structured_aux.paired_outcome_preference_every_updates) > 0
        )

    @property
    def structured_warmstart_enabled(self) -> bool:
        return bool(self.structured_warmstart.enabled) and int(self.structured_warmstart.updates) > 0


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
