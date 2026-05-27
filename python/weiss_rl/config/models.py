"""Immutable grouped config models for the RL presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    role: str


@dataclass(frozen=True, slots=True)
class SystemProfileConfig:
    training: str
    local_iteration: str
    ci_invariant_testing: str


@dataclass(frozen=True, slots=True)
class SystemConfig:
    profile: SystemProfileConfig
    mp_start_method: str
    learner_device: str
    actor_device: str
    actor_process_count: int
    envs_per_actor: int
    total_envs: int
    actor_torch_threads: int
    learner_torch_threads: int
    actor_queue_capacity_unrolls: int
    learner_prefetch_batches: int
    collection_backend: str = "auto"


@dataclass(frozen=True, slots=True)
class ModelDropoutConfig:
    family_a: float
    ablation: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    gru_hidden_size: int
    encoder_mlp_width: int
    encoder_mlp_layers: int
    layer_norm: bool
    dropout: ModelDropoutConfig
    encoder_kind: str = "mlp"
    structured_policy_contract: str = "packed_v1"
    typed_feature_width: int = 64
    recurrent_core: str = "gru"
    candidate_scoring_chunk_size: int = 65536
    cuda_learner_candidate_scoring_chunk_size: int = 262144
    public_heuristic_logit_bias_scale: float = 0.0
    public_heuristic_actor_logit_bias_scale: float = -1.0
    public_heuristic_logit_bias_start_updates: int = 0
    public_heuristic_logit_bias_end_updates: int = -1
    public_heuristic_logit_bias_final_scale: float = 0.0
    public_heuristic_logit_bias_families: tuple[str, ...] = field(default_factory=tuple)
    opponent_context_policy_ids: tuple[str, ...] = field(default_factory=tuple)
    opponent_context_hidden_scale: float = 0.0
    opponent_context_trainable_hidden_scale: float = 0.0
    opponent_context_trainable_recurrent_scale: float = 0.0
    opponent_context_trainable_action_bias_scale: float = 0.0
    opponent_context_trainable_candidate_residual_scale: float = 0.0
    opponent_context_candidate_residual_width: int = 32
    opponent_context_candidate_residual_mode: str = "additive"
    opponent_context_candidate_residual_action_ids: tuple[int, ...] = field(default_factory=tuple)
    opponent_context_adapter_lr_multiplier: float = 1.0
    opponent_context_adapter_train_only: bool = False
    opponent_context_eval_policy_ids: tuple[str, ...] = field(default_factory=tuple)


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


@dataclass(frozen=True, slots=True)
class StopRulesConfig:
    stop_delta_ci_half_width: float
    stop_confidence: float


@dataclass(frozen=True, slots=True)
class LegalFingerprintChecksConfig:
    enabled: bool
    version: str
    require_strictly_increasing_legal_ids: bool
    mismatch_policy: str


@dataclass(frozen=True, slots=True)
class DecisionKindTaggingConfig:
    required_for_training: bool
    enable_python_derived_debug_tag: bool


@dataclass(frozen=True, slots=True)
class FixedAnchorSetConfig:
    required: tuple[str, ...]
    optional_if_available: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalPolicySetSelectionConfig:
    version: str
    include_random_legal_baseline_b0: bool
    include_no_league_baseline_b1: bool
    include_heuristic_public_b2_if_exists: bool
    include_heuristic_public_anchors_b2_b3_b4: bool
    include_final_champion_snapshot: bool
    include_spaced_snapshots_near_percent_updates: tuple[int, ...]
    remaining_slots_strategy: str
    fixed_anchor_set_v1: FixedAnchorSetConfig
    seed_file: str
    folding: str
    seat_swap: bool
    tie_break: str


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seat_swap: bool
    eval_device: str
    eval_inference_mode: bool
    eval_sampling_algorithm: str
    eval_assert_sorted_legal_ids: bool
    seed_files: dict[str, str]
    periodic_dev_eval_interval_updates: int
    periodic_dev_eval_paired_seeds: int
    final_policy_set_size: int
    final_matrix_stage1_paired_seeds: int
    final_matrix_stage2_adaptive_max_paired_seeds: int
    stop_rules: StopRulesConfig
    replay_capture_rate_eval: float
    regression_capture_count: int
    legal_fingerprint_checks: LegalFingerprintChecksConfig
    decision_kind_tagging: DecisionKindTaggingConfig
    final_policy_set_selection: FinalPolicySetSelectionConfig
    model_sampling_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class SpecBundlePolicyConfig:
    require_export_spec_bundle: bool
    persist_in_manifest: bool
    fail_on_spec_mismatch: bool


@dataclass(frozen=True, slots=True)
class IdsConfig:
    run_id_hash: str
    config_hash: str
    spec_hash: str
    store_full_256_bit_ids: bool
    store_short_64_bit_ids_for_filenames: bool


@dataclass(frozen=True, slots=True)
class SeedDerivationConfig:
    base_seed64: int
    actor_seed_formula: str
    episode_seed_formula: str


@dataclass(frozen=True, slots=True)
class LegalFingerprintConfig:
    version: str
    compute_in_rl_layer: bool
    canonical_bytes: tuple[str, ...]
    replay_eval_mismatch_policy: str


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    spec_bundle: SpecBundlePolicyConfig
    ids: IdsConfig
    seed_derivation: SeedDerivationConfig
    seed_files: dict[str, str]
    determinism_requirements: tuple[str, ...]
    legal_fingerprint: LegalFingerprintConfig


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
