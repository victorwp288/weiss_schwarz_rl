"""Immutable config models for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, slots=True)
class TrainingFamilyAConfig:
    algorithm: str
    unroll_length: int
    batch_unrolls_per_update: int
    gamma: float
    reward_mode: str
    optimizer: str
    learning_rate: float
    grad_norm_clip: float
    value_loss_coef: float
    entropy_coef: float
    entropy_anneal_to: float
    entropy_anneal_steps_updates: int
    vtrace_rho_bar: float
    vtrace_c_bar: float
    mixed_precision: bool
    masking_math_float32: bool
    checkpoint_interval_updates: int
    snapshot_interval_updates: int


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
    truncation_reward: float
    shaping_enabled_family_a: bool
    deck_set_size: DeckSetSizeConfig
    truncation_bootstrap_value: bool
    truncation_bootstrap_rule: str


@dataclass(frozen=True, slots=True)
class LeagueWarmupConfig:
    first_updates: int
    initial_window_episodes: int
    ramp_target_updates: int
    ramp_target_window_episodes: int


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
class LeagueConfig:
    enabled: bool
    snapshot_pool_recent_size: int
    snapshot_pool_champion_size: int
    opponent_sampling: str
    pfsp_power: float
    pfsp_epsilon_uniform: float
    pfsp_stats_source: str
    pfsp_window_episodes: int
    warmup: LeagueWarmupConfig
    promotion_gate_enabled: bool
    promotion_gate_paired_seeds: int
    promotion_threshold: str
    promotion_anchor_set_v1: PromotionAnchorSetConfig
    promotion_seed_file: str
    promotion_gate: PromotionGateConfig


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
class NashConfig:
    impl: str
    backend: str
    threads: int
    value_tolerance: float
    tie_break: str


@dataclass(frozen=True, slots=True)
class AlphaRankConfig:
    impl: str
    m: int
    alpha: int
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
    nash: NashConfig
    alpharank: AlphaRankConfig


@dataclass(frozen=True, slots=True)
class SensitivityCaseConfig:
    draw_score: float
    description: str
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
class ComputeBudgetAllocationConfig:
    bring_up_correctness: int
    main_training_3_seeds: int
    ablations: int
    baseline_extra_run: int
    reserve: int


@dataclass(frozen=True, slots=True)
class ComputeBudgetConfig:
    baseline_credits: int
    allocation: ComputeBudgetAllocationConfig
    calibration_required: bool
    calibration_metrics: tuple[str, ...]
    update_targets_from_calibration: bool
    drift_alert_threshold_percent: int


@dataclass(frozen=True, slots=True)
class FamilyBDiscountAblationConfig:
    enabled_by_default: bool
    gamma_default: float
    requires_k_raw_decisions_tracking: bool
    gamma_step_formula: str


@dataclass(frozen=True, slots=True)
class FamilyCStallTriggerConfig:
    after_updates: int
    eval_opponent: str
    eval_seed_file: str
    seat_swap: bool
    probability_metric: str
    trigger_if_below: float


@dataclass(frozen=True, slots=True)
class FamilyCShapingDefaultsConfig:
    terminal_win: float
    terminal_loss: float
    terminal_draw_timeout: float
    per_learner_step_penalty_formula: str
    lambda_default: float
    max_total_shaping_magnitude_per_episode: float


@dataclass(frozen=True, slots=True)
class FamilyCShapingAblationConfig:
    enabled_by_default: bool
    stall_trigger: FamilyCStallTriggerConfig
    shaping_defaults: FamilyCShapingDefaultsConfig
    truncation_reward: float


@dataclass(frozen=True, slots=True)
class LockedConfig:
    system: SystemConfig | None = None
    model: ModelConfig | None = None
    training_family_a: TrainingFamilyAConfig | None = None
    environment: EnvironmentConfig | None = None
    league: LeagueConfig | None = None
    evaluation: EvaluationConfig | None = None
    reproducibility: ReproducibilityConfig | None = None
    metagame: MetagameConfig | None = None
    sensitivity: SensitivityConfig | None = None
    compute_budget: ComputeBudgetConfig | None = None
    family_b_discount_ablation: FamilyBDiscountAblationConfig | None = None
    family_c_shaping_ablation: FamilyCShapingAblationConfig | None = None


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
