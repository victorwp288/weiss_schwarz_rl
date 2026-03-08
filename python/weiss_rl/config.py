"""Config loading utilities for the RL stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_str_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_require_text(item, field_name=f"{field_name}[]") for item in value)


def _resolve_repo_path(root: Path, relative_path: str) -> Path:
    return (root / relative_path).resolve()


def _load_component_doc(path: Path, component_name: str) -> dict[str, Any]:
    doc = _load_yaml(path)
    body = doc.get(component_name, doc)
    return _require_mapping(body, context=component_name)


def _parse_system_config(body: dict[str, Any]) -> SystemConfig:
    profile = _require_mapping(body["profile"], context="system.profile")
    return SystemConfig(
        profile=SystemProfileConfig(
            training=_require_text(profile["training"], field_name="system.profile.training"),
            local_iteration=_require_text(profile["local_iteration"], field_name="system.profile.local_iteration"),
            ci_invariant_testing=_require_text(profile["ci_invariant_testing"], field_name="system.profile.ci_invariant_testing"),
        ),
        mp_start_method=_require_text(body["mp_start_method"], field_name="system.mp_start_method"),
        learner_device=_require_text(body["learner_device"], field_name="system.learner_device"),
        actor_device=_require_text(body["actor_device"], field_name="system.actor_device"),
        actor_process_count=_require_int(body["actor_process_count"], field_name="system.actor_process_count", minimum=1),
        envs_per_actor=_require_int(body["envs_per_actor"], field_name="system.envs_per_actor", minimum=1),
        total_envs=_require_int(body["total_envs"], field_name="system.total_envs", minimum=1),
        actor_torch_threads=_require_int(body["actor_torch_threads"], field_name="system.actor_torch_threads", minimum=1),
        learner_torch_threads=_require_int(body["learner_torch_threads"], field_name="system.learner_torch_threads", minimum=1),
        actor_queue_capacity_unrolls=_require_int(
            body["actor_queue_capacity_unrolls"],
            field_name="system.actor_queue_capacity_unrolls",
            minimum=1,
        ),
        learner_prefetch_batches=_require_int(
            body["learner_prefetch_batches"],
            field_name="system.learner_prefetch_batches",
            minimum=1,
        ),
    )


def _parse_model_config(body: dict[str, Any]) -> ModelConfig:
    dropout = _require_mapping(body["dropout"], context="model.dropout")
    return ModelConfig(
        gru_hidden_size=_require_int(body["gru_hidden_size"], field_name="model.gru_hidden_size", minimum=1),
        encoder_mlp_width=_require_int(body["encoder_mlp_width"], field_name="model.encoder_mlp_width", minimum=1),
        encoder_mlp_layers=_require_int(body["encoder_mlp_layers"], field_name="model.encoder_mlp_layers", minimum=1),
        layer_norm=_require_bool(body["layer_norm"], field_name="model.layer_norm"),
        dropout=ModelDropoutConfig(
            family_a=_require_float(dropout["family_a"], field_name="model.dropout.family_a"),
            ablation=_require_float(dropout["ablation"], field_name="model.dropout.ablation"),
        ),
    )


def _parse_training_family_a_config(body: dict[str, Any]) -> TrainingFamilyAConfig:
    return TrainingFamilyAConfig(
        algorithm=_require_text(body["algorithm"], field_name="training_family_a.algorithm"),
        unroll_length=_require_int(body["unroll_length"], field_name="training_family_a.unroll_length", minimum=1),
        batch_unrolls_per_update=_require_int(
            body["batch_unrolls_per_update"],
            field_name="training_family_a.batch_unrolls_per_update",
            minimum=1,
        ),
        gamma=_require_float(body["gamma"], field_name="training_family_a.gamma"),
        reward_mode=_require_text(body["reward_mode"], field_name="training_family_a.reward_mode"),
        optimizer=_require_text(body["optimizer"], field_name="training_family_a.optimizer"),
        learning_rate=_require_float(body["learning_rate"], field_name="training_family_a.learning_rate"),
        grad_norm_clip=_require_float(body["grad_norm_clip"], field_name="training_family_a.grad_norm_clip"),
        value_loss_coef=_require_float(body["value_loss_coef"], field_name="training_family_a.value_loss_coef"),
        entropy_coef=_require_float(body["entropy_coef"], field_name="training_family_a.entropy_coef"),
        entropy_anneal_to=_require_float(body["entropy_anneal_to"], field_name="training_family_a.entropy_anneal_to"),
        entropy_anneal_steps_updates=_require_int(
            body["entropy_anneal_steps_updates"],
            field_name="training_family_a.entropy_anneal_steps_updates",
            minimum=1,
        ),
        vtrace_rho_bar=_require_float(body["vtrace_rho_bar"], field_name="training_family_a.vtrace_rho_bar"),
        vtrace_c_bar=_require_float(body["vtrace_c_bar"], field_name="training_family_a.vtrace_c_bar"),
        mixed_precision=_require_bool(body["mixed_precision"], field_name="training_family_a.mixed_precision"),
        masking_math_float32=_require_bool(
            body["masking_math_float32"],
            field_name="training_family_a.masking_math_float32",
        ),
        checkpoint_interval_updates=_require_int(
            body["checkpoint_interval_updates"],
            field_name="training_family_a.checkpoint_interval_updates",
            minimum=1,
        ),
        snapshot_interval_updates=_require_int(
            body["snapshot_interval_updates"],
            field_name="training_family_a.snapshot_interval_updates",
            minimum=1,
        ),
    )


def _parse_environment_config(body: dict[str, Any]) -> EnvironmentConfig:
    deck_set_size = _require_mapping(body["deck_set_size"], context="environment.deck_set_size")
    return EnvironmentConfig(
        observation_visibility=_require_text(body["observation_visibility"], field_name="environment.observation_visibility"),
        visibility=_require_text(body["visibility"], field_name="environment.visibility"),
        truncate_on_max_steps=_require_bool(body["truncate_on_max_steps"], field_name="environment.truncate_on_max_steps"),
        max_raw_decisions_per_episode=_require_int(
            body["max_raw_decisions_per_episode"],
            field_name="environment.max_raw_decisions_per_episode",
            minimum=1,
        ),
        max_decisions=_require_int(body["max_decisions"], field_name="environment.max_decisions", minimum=1),
        max_decisions_per_episode=_require_int(
            body["max_decisions_per_episode"],
            field_name="environment.max_decisions_per_episode",
            minimum=1,
        ),
        max_learner_steps_per_episode=_require_int(
            body["max_learner_steps_per_episode"],
            field_name="environment.max_learner_steps_per_episode",
            minimum=1,
        ),
        max_ticks=_require_int(body["max_ticks"], field_name="environment.max_ticks", minimum=1),
        truncation_reward=_require_float(body["truncation_reward"], field_name="environment.truncation_reward"),
        shaping_enabled_family_a=_require_bool(
            body["shaping_enabled_family_a"],
            field_name="environment.shaping_enabled_family_a",
        ),
        deck_set_size=DeckSetSizeConfig(
            bring_up=_require_int(deck_set_size["bring_up"], field_name="environment.deck_set_size.bring_up", minimum=1),
            paper=_require_int(deck_set_size["paper"], field_name="environment.deck_set_size.paper", minimum=1),
        ),
        truncation_bootstrap_value=_require_bool(
            body["truncation_bootstrap_value"],
            field_name="environment.truncation_bootstrap_value",
        ),
        truncation_bootstrap_rule=_require_text(
            body["truncation_bootstrap_rule"],
            field_name="environment.truncation_bootstrap_rule",
        ),
    )


def _parse_league_config(body: dict[str, Any]) -> LeagueConfig:
    warmup = _require_mapping(body["warmup"], context="league.warmup")
    anchor_set = _require_mapping(body["promotion_anchor_set_v1"], context="league.promotion_anchor_set_v1")
    promotion_gate = _require_mapping(body["promotion_gate"], context="league.promotion_gate")
    guardrails = _require_mapping(promotion_gate["guardrails"], context="league.promotion_gate.guardrails")
    return LeagueConfig(
        enabled=_require_bool(body["enabled"], field_name="league.enabled"),
        snapshot_pool_recent_size=_require_int(
            body["snapshot_pool_recent_size"],
            field_name="league.snapshot_pool_recent_size",
            minimum=1,
        ),
        snapshot_pool_champion_size=_require_int(
            body["snapshot_pool_champion_size"],
            field_name="league.snapshot_pool_champion_size",
            minimum=1,
        ),
        opponent_sampling=_require_text(body["opponent_sampling"], field_name="league.opponent_sampling"),
        pfsp_power=_require_float(body["pfsp_power"], field_name="league.pfsp_power"),
        pfsp_epsilon_uniform=_require_float(
            body["pfsp_epsilon_uniform"],
            field_name="league.pfsp_epsilon_uniform",
        ),
        pfsp_stats_source=_require_text(body["pfsp_stats_source"], field_name="league.pfsp_stats_source"),
        pfsp_window_episodes=_require_int(
            body["pfsp_window_episodes"],
            field_name="league.pfsp_window_episodes",
            minimum=1,
        ),
        warmup=LeagueWarmupConfig(
            first_updates=_require_int(warmup["first_updates"], field_name="league.warmup.first_updates", minimum=1),
            initial_window_episodes=_require_int(
                warmup["initial_window_episodes"],
                field_name="league.warmup.initial_window_episodes",
                minimum=1,
            ),
            ramp_target_updates=_require_int(
                warmup["ramp_target_updates"],
                field_name="league.warmup.ramp_target_updates",
                minimum=1,
            ),
            ramp_target_window_episodes=_require_int(
                warmup["ramp_target_window_episodes"],
                field_name="league.warmup.ramp_target_window_episodes",
                minimum=1,
            ),
        ),
        promotion_gate_enabled=_require_bool(body["promotion_gate_enabled"], field_name="league.promotion_gate_enabled"),
        promotion_gate_paired_seeds=_require_int(
            body["promotion_gate_paired_seeds"],
            field_name="league.promotion_gate_paired_seeds",
            minimum=1,
        ),
        promotion_threshold=_require_text(body["promotion_threshold"], field_name="league.promotion_threshold"),
        promotion_anchor_set_v1=PromotionAnchorSetConfig(
            required=_require_str_list(anchor_set["required"], field_name="league.promotion_anchor_set_v1.required"),
            optional_if_available=_require_str_list(
                anchor_set["optional_if_available"],
                field_name="league.promotion_anchor_set_v1.optional_if_available",
            ),
        ),
        promotion_seed_file=_require_text(body["promotion_seed_file"], field_name="league.promotion_seed_file"),
        promotion_gate=PromotionGateConfig(
            uncertainty_method=_require_text(
                promotion_gate["uncertainty_method"],
                field_name="league.promotion_gate.uncertainty_method",
            ),
            weighting=_require_text(promotion_gate["weighting"], field_name="league.promotion_gate.weighting"),
            seat_swap=_require_bool(promotion_gate["seat_swap"], field_name="league.promotion_gate.seat_swap"),
            folding=_require_text(promotion_gate["folding"], field_name="league.promotion_gate.folding"),
            guardrails=PromotionGateGuardrailsConfig(
                max_prob_anchor_loss_below_0_45=_require_float(
                    guardrails["max_prob_anchor_loss_below_0_45"],
                    field_name="league.promotion_gate.guardrails.max_prob_anchor_loss_below_0_45",
                ),
                max_truncation_rate=_require_float(
                    guardrails["max_truncation_rate"],
                    field_name="league.promotion_gate.guardrails.max_truncation_rate",
                ),
            ),
            record_file=_require_text(promotion_gate["record_file"], field_name="league.promotion_gate.record_file"),
        ),
    )


def _parse_evaluation_config(body: dict[str, Any]) -> EvaluationConfig:
    stop_rules = _require_mapping(body["stop_rules"], context="evaluation.stop_rules")
    legal_fingerprint_checks = _require_mapping(
        body["legal_fingerprint_checks"],
        context="evaluation.legal_fingerprint_checks",
    )
    decision_kind_tagging = _require_mapping(body["decision_kind_tagging"], context="evaluation.decision_kind_tagging")
    final_policy_set_selection = _require_mapping(
        body["final_policy_set_selection"],
        context="evaluation.final_policy_set_selection",
    )
    fixed_anchor_set = _require_mapping(
        final_policy_set_selection["fixed_anchor_set_v1"],
        context="evaluation.final_policy_set_selection.fixed_anchor_set_v1",
    )
    seed_files = _require_mapping(body["seed_files"], context="evaluation.seed_files")
    return EvaluationConfig(
        seat_swap=_require_bool(body["seat_swap"], field_name="evaluation.seat_swap"),
        eval_device=_require_text(body["eval_device"], field_name="evaluation.eval_device"),
        eval_inference_mode=_require_bool(body["eval_inference_mode"], field_name="evaluation.eval_inference_mode"),
        eval_sampling_algorithm=_require_text(
            body["eval_sampling_algorithm"],
            field_name="evaluation.eval_sampling_algorithm",
        ),
        eval_assert_sorted_legal_ids=_require_bool(
            body["eval_assert_sorted_legal_ids"],
            field_name="evaluation.eval_assert_sorted_legal_ids",
        ),
        seed_files={
            key: _require_text(value, field_name=f"evaluation.seed_files.{key}")
            for key, value in seed_files.items()
        },
        periodic_dev_eval_interval_updates=_require_int(
            body["periodic_dev_eval_interval_updates"],
            field_name="evaluation.periodic_dev_eval_interval_updates",
            minimum=1,
        ),
        periodic_dev_eval_paired_seeds=_require_int(
            body["periodic_dev_eval_paired_seeds"],
            field_name="evaluation.periodic_dev_eval_paired_seeds",
            minimum=1,
        ),
        final_policy_set_size=_require_int(
            body["final_policy_set_size"],
            field_name="evaluation.final_policy_set_size",
            minimum=1,
        ),
        final_matrix_stage1_paired_seeds=_require_int(
            body["final_matrix_stage1_paired_seeds"],
            field_name="evaluation.final_matrix_stage1_paired_seeds",
            minimum=1,
        ),
        final_matrix_stage2_adaptive_max_paired_seeds=_require_int(
            body["final_matrix_stage2_adaptive_max_paired_seeds"],
            field_name="evaluation.final_matrix_stage2_adaptive_max_paired_seeds",
            minimum=1,
        ),
        stop_rules=StopRulesConfig(
            stop_delta_ci_half_width=_require_float(
                stop_rules["stop_delta_ci_half_width"],
                field_name="evaluation.stop_rules.stop_delta_ci_half_width",
            ),
            stop_confidence=_require_float(
                stop_rules["stop_confidence"],
                field_name="evaluation.stop_rules.stop_confidence",
            ),
        ),
        replay_capture_rate_eval=_require_float(
            body["replay_capture_rate_eval"],
            field_name="evaluation.replay_capture_rate_eval",
        ),
        regression_capture_count=_require_int(
            body["regression_capture_count"],
            field_name="evaluation.regression_capture_count",
            minimum=1,
        ),
        legal_fingerprint_checks=LegalFingerprintChecksConfig(
            enabled=_require_bool(
                legal_fingerprint_checks["enabled"],
                field_name="evaluation.legal_fingerprint_checks.enabled",
            ),
            version=_require_text(
                legal_fingerprint_checks["version"],
                field_name="evaluation.legal_fingerprint_checks.version",
            ),
            require_strictly_increasing_legal_ids=_require_bool(
                legal_fingerprint_checks["require_strictly_increasing_legal_ids"],
                field_name="evaluation.legal_fingerprint_checks.require_strictly_increasing_legal_ids",
            ),
            mismatch_policy=_require_text(
                legal_fingerprint_checks["mismatch_policy"],
                field_name="evaluation.legal_fingerprint_checks.mismatch_policy",
            ),
        ),
        decision_kind_tagging=DecisionKindTaggingConfig(
            required_for_training=_require_bool(
                decision_kind_tagging["required_for_training"],
                field_name="evaluation.decision_kind_tagging.required_for_training",
            ),
            enable_python_derived_debug_tag=_require_bool(
                decision_kind_tagging["enable_python_derived_debug_tag"],
                field_name="evaluation.decision_kind_tagging.enable_python_derived_debug_tag",
            ),
        ),
        final_policy_set_selection=FinalPolicySetSelectionConfig(
            version=_require_text(
                final_policy_set_selection["version"],
                field_name="evaluation.final_policy_set_selection.version",
            ),
            include_random_legal_baseline_b0=_require_bool(
                final_policy_set_selection["include_random_legal_baseline_b0"],
                field_name="evaluation.final_policy_set_selection.include_random_legal_baseline_b0",
            ),
            include_no_league_baseline_b1=_require_bool(
                final_policy_set_selection["include_no_league_baseline_b1"],
                field_name="evaluation.final_policy_set_selection.include_no_league_baseline_b1",
            ),
            include_heuristic_public_b2_if_exists=_require_bool(
                final_policy_set_selection["include_heuristic_public_b2_if_exists"],
                field_name="evaluation.final_policy_set_selection.include_heuristic_public_b2_if_exists",
            ),
            include_final_champion_snapshot=_require_bool(
                final_policy_set_selection["include_final_champion_snapshot"],
                field_name="evaluation.final_policy_set_selection.include_final_champion_snapshot",
            ),
            include_spaced_snapshots_near_percent_updates=tuple(
                _require_int(
                    item,
                    field_name="evaluation.final_policy_set_selection.include_spaced_snapshots_near_percent_updates[]",
                    minimum=0,
                )
                for item in final_policy_set_selection["include_spaced_snapshots_near_percent_updates"]
            ),
            remaining_slots_strategy=_require_text(
                final_policy_set_selection["remaining_slots_strategy"],
                field_name="evaluation.final_policy_set_selection.remaining_slots_strategy",
            ),
            fixed_anchor_set_v1=FixedAnchorSetConfig(
                required=_require_str_list(
                    fixed_anchor_set["required"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.required",
                ),
                optional_if_available=_require_str_list(
                    fixed_anchor_set["optional_if_available"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available",
                ),
            ),
            seed_file=_require_text(
                final_policy_set_selection["seed_file"],
                field_name="evaluation.final_policy_set_selection.seed_file",
            ),
            folding=_require_text(
                final_policy_set_selection["folding"],
                field_name="evaluation.final_policy_set_selection.folding",
            ),
            seat_swap=_require_bool(
                final_policy_set_selection["seat_swap"],
                field_name="evaluation.final_policy_set_selection.seat_swap",
            ),
            tie_break=_require_text(
                final_policy_set_selection["tie_break"],
                field_name="evaluation.final_policy_set_selection.tie_break",
            ),
        ),
    )


def _parse_reproducibility_config(body: dict[str, Any]) -> ReproducibilityConfig:
    spec_bundle = _require_mapping(body["spec_bundle"], context="reproducibility.spec_bundle")
    ids = _require_mapping(body["ids"], context="reproducibility.ids")
    seed_derivation = _require_mapping(body["seed_derivation"], context="reproducibility.seed_derivation")
    seed_files = _require_mapping(body["seed_files"], context="reproducibility.seed_files")
    legal_fingerprint = _require_mapping(body["legal_fingerprint"], context="reproducibility.legal_fingerprint")
    return ReproducibilityConfig(
        spec_bundle=SpecBundlePolicyConfig(
            require_export_spec_bundle=_require_bool(
                spec_bundle["require_export_spec_bundle"],
                field_name="reproducibility.spec_bundle.require_export_spec_bundle",
            ),
            persist_in_manifest=_require_bool(
                spec_bundle["persist_in_manifest"],
                field_name="reproducibility.spec_bundle.persist_in_manifest",
            ),
            fail_on_spec_mismatch=_require_bool(
                spec_bundle["fail_on_spec_mismatch"],
                field_name="reproducibility.spec_bundle.fail_on_spec_mismatch",
            ),
        ),
        ids=IdsConfig(
            run_id_hash=_require_text(ids["run_id_hash"], field_name="reproducibility.ids.run_id_hash"),
            config_hash=_require_text(ids["config_hash"], field_name="reproducibility.ids.config_hash"),
            spec_hash=_require_text(ids["spec_hash"], field_name="reproducibility.ids.spec_hash"),
            store_full_256_bit_ids=_require_bool(
                ids["store_full_256_bit_ids"],
                field_name="reproducibility.ids.store_full_256_bit_ids",
            ),
            store_short_64_bit_ids_for_filenames=_require_bool(
                ids["store_short_64_bit_ids_for_filenames"],
                field_name="reproducibility.ids.store_short_64_bit_ids_for_filenames",
            ),
        ),
        seed_derivation=SeedDerivationConfig(
            base_seed64=_require_int(
                seed_derivation["base_seed64"],
                field_name="reproducibility.seed_derivation.base_seed64",
                minimum=0,
            ),
            actor_seed_formula=_require_text(
                seed_derivation["actor_seed_formula"],
                field_name="reproducibility.seed_derivation.actor_seed_formula",
            ),
            episode_seed_formula=_require_text(
                seed_derivation["episode_seed_formula"],
                field_name="reproducibility.seed_derivation.episode_seed_formula",
            ),
        ),
        seed_files={
            key: _require_text(value, field_name=f"reproducibility.seed_files.{key}")
            for key, value in seed_files.items()
        },
        determinism_requirements=_require_str_list(
            body["determinism_requirements"],
            field_name="reproducibility.determinism_requirements",
        ),
        legal_fingerprint=LegalFingerprintConfig(
            version=_require_text(
                legal_fingerprint["version"],
                field_name="reproducibility.legal_fingerprint.version",
            ),
            compute_in_rl_layer=_require_bool(
                legal_fingerprint["compute_in_rl_layer"],
                field_name="reproducibility.legal_fingerprint.compute_in_rl_layer",
            ),
            canonical_bytes=_require_str_list(
                legal_fingerprint["canonical_bytes"],
                field_name="reproducibility.legal_fingerprint.canonical_bytes",
            ),
            replay_eval_mismatch_policy=_require_text(
                legal_fingerprint["replay_eval_mismatch_policy"],
                field_name="reproducibility.legal_fingerprint.replay_eval_mismatch_policy",
            ),
        ),
    )


def _parse_metagame_config(body: dict[str, Any]) -> MetagameConfig:
    nash = _require_mapping(body["nash"], context="metagame.nash")
    alpharank = _require_mapping(body["alpharank"], context="metagame.alpharank")
    return MetagameConfig(
        payoff_uncertainty_method=_require_text(
            body["payoff_uncertainty_method"],
            field_name="metagame.payoff_uncertainty_method",
        ),
        sampling_m=_require_int(body["sampling_M"], field_name="metagame.sampling_M", minimum=1),
        optional_secondary_uncertainty_method=_require_text(
            body["optional_secondary_uncertainty_method"],
            field_name="metagame.optional_secondary_uncertainty_method",
        ),
        dirichlet_alpha_wldt=_require_float(
            body["dirichlet_alpha_wldt"],
            field_name="metagame.dirichlet_alpha_wldt",
        ),
        primary_analysis=_require_text(body["primary_analysis"], field_name="metagame.primary_analysis"),
        secondary_analysis=_require_text(body["secondary_analysis"], field_name="metagame.secondary_analysis"),
        nash=NashConfig(
            impl=_require_text(nash["impl"], field_name="metagame.nash.impl"),
            backend=_require_text(nash["backend"], field_name="metagame.nash.backend"),
            threads=_require_int(nash["threads"], field_name="metagame.nash.threads", minimum=1),
            value_tolerance=_require_float(nash["value_tolerance"], field_name="metagame.nash.value_tolerance"),
            tie_break=_require_text(nash["tie_break"], field_name="metagame.nash.tie_break"),
        ),
        alpharank=AlphaRankConfig(
            impl=_require_text(alpharank["impl"], field_name="metagame.alpharank.impl"),
            m=_require_int(alpharank["m"], field_name="metagame.alpharank.m", minimum=1),
            alpha=_require_int(alpharank["alpha"], field_name="metagame.alpharank.alpha", minimum=1),
            local_selection=_require_bool(
                alpharank["local_selection"],
                field_name="metagame.alpharank.local_selection",
            ),
            use_inf_alpha=_require_bool(alpharank["use_inf_alpha"], field_name="metagame.alpharank.use_inf_alpha"),
            inf_alpha_eps=_require_float(alpharank["inf_alpha_eps"], field_name="metagame.alpharank.inf_alpha_eps"),
        ),
    )


def _parse_sensitivity_config(body: dict[str, Any]) -> SensitivityConfig:
    report = _require_mapping(body["report"], context="sensitivity.report")
    cases: dict[str, SensitivityCaseConfig] = {}
    for key, value in body.items():
        if key == "report":
            continue
        case = _require_mapping(value, context=f"sensitivity.{key}")
        cases[key] = SensitivityCaseConfig(
            draw_score=_require_float(case["draw_score"], field_name=f"sensitivity.{key}.draw_score"),
            description=_require_text(case["description"], field_name=f"sensitivity.{key}.description"),
            truncation_score=(
                _require_float(case["truncation_score"], field_name=f"sensitivity.{key}.truncation_score")
                if "truncation_score" in case
                else None
            ),
            truncation_handling=(
                _require_text(case["truncation_handling"], field_name=f"sensitivity.{key}.truncation_handling")
                if "truncation_handling" in case
                else None
            ),
        )
    return SensitivityConfig(
        cases=cases,
        report=SensitivityReportConfig(
            required_outputs=_require_str_list(report["required_outputs"], field_name="sensitivity.report.required_outputs")
        ),
    )


def _parse_compute_budget_config(body: dict[str, Any]) -> ComputeBudgetConfig:
    allocation = _require_mapping(body["allocation"], context="compute_budget.allocation")
    return ComputeBudgetConfig(
        baseline_credits=_require_int(body["baseline_credits"], field_name="compute_budget.baseline_credits", minimum=1),
        allocation=ComputeBudgetAllocationConfig(
            bring_up_correctness=_require_int(
                allocation["bring_up_correctness"],
                field_name="compute_budget.allocation.bring_up_correctness",
                minimum=0,
            ),
            main_training_3_seeds=_require_int(
                allocation["main_training_3_seeds"],
                field_name="compute_budget.allocation.main_training_3_seeds",
                minimum=0,
            ),
            ablations=_require_int(allocation["ablations"], field_name="compute_budget.allocation.ablations", minimum=0),
            baseline_extra_run=_require_int(
                allocation["baseline_extra_run"],
                field_name="compute_budget.allocation.baseline_extra_run",
                minimum=0,
            ),
            reserve=_require_int(allocation["reserve"], field_name="compute_budget.allocation.reserve", minimum=0),
        ),
        calibration_required=_require_bool(body["calibration_required"], field_name="compute_budget.calibration_required"),
        calibration_metrics=_require_str_list(body["calibration_metrics"], field_name="compute_budget.calibration_metrics"),
        update_targets_from_calibration=_require_bool(
            body["update_targets_from_calibration"],
            field_name="compute_budget.update_targets_from_calibration",
        ),
        drift_alert_threshold_percent=_require_int(
            body["drift_alert_threshold_percent"],
            field_name="compute_budget.drift_alert_threshold_percent",
            minimum=0,
        ),
    )


def _parse_family_b_discount_ablation_config(body: dict[str, Any]) -> FamilyBDiscountAblationConfig:
    return FamilyBDiscountAblationConfig(
        enabled_by_default=_require_bool(body["enabled_by_default"], field_name="family_b_discount_ablation.enabled_by_default"),
        gamma_default=_require_float(body["gamma_default"], field_name="family_b_discount_ablation.gamma_default"),
        requires_k_raw_decisions_tracking=_require_bool(
            body["requires_k_raw_decisions_tracking"],
            field_name="family_b_discount_ablation.requires_k_raw_decisions_tracking",
        ),
        gamma_step_formula=_require_text(
            body["gamma_step_formula"],
            field_name="family_b_discount_ablation.gamma_step_formula",
        ),
    )


def _parse_family_c_shaping_ablation_config(body: dict[str, Any]) -> FamilyCShapingAblationConfig:
    stall_trigger = _require_mapping(body["stall_trigger"], context="family_c_shaping_ablation.stall_trigger")
    shaping_defaults = _require_mapping(body["shaping_defaults"], context="family_c_shaping_ablation.shaping_defaults")
    return FamilyCShapingAblationConfig(
        enabled_by_default=_require_bool(body["enabled_by_default"], field_name="family_c_shaping_ablation.enabled_by_default"),
        stall_trigger=FamilyCStallTriggerConfig(
            after_updates=_require_int(
                stall_trigger["after_updates"],
                field_name="family_c_shaping_ablation.stall_trigger.after_updates",
                minimum=1,
            ),
            eval_opponent=_require_text(
                stall_trigger["eval_opponent"],
                field_name="family_c_shaping_ablation.stall_trigger.eval_opponent",
            ),
            eval_seed_file=_require_text(
                stall_trigger["eval_seed_file"],
                field_name="family_c_shaping_ablation.stall_trigger.eval_seed_file",
            ),
            seat_swap=_require_bool(
                stall_trigger["seat_swap"],
                field_name="family_c_shaping_ablation.stall_trigger.seat_swap",
            ),
            probability_metric=_require_text(
                stall_trigger["probability_metric"],
                field_name="family_c_shaping_ablation.stall_trigger.probability_metric",
            ),
            trigger_if_below=_require_float(
                stall_trigger["trigger_if_below"],
                field_name="family_c_shaping_ablation.stall_trigger.trigger_if_below",
            ),
        ),
        shaping_defaults=FamilyCShapingDefaultsConfig(
            terminal_win=_require_float(
                shaping_defaults["terminal_win"],
                field_name="family_c_shaping_ablation.shaping_defaults.terminal_win",
            ),
            terminal_loss=_require_float(
                shaping_defaults["terminal_loss"],
                field_name="family_c_shaping_ablation.shaping_defaults.terminal_loss",
            ),
            terminal_draw_timeout=_require_float(
                shaping_defaults["terminal_draw_timeout"],
                field_name="family_c_shaping_ablation.shaping_defaults.terminal_draw_timeout",
            ),
            per_learner_step_penalty_formula=_require_text(
                shaping_defaults["per_learner_step_penalty_formula"],
                field_name="family_c_shaping_ablation.shaping_defaults.per_learner_step_penalty_formula",
            ),
            lambda_default=_require_float(
                shaping_defaults["lambda_default"],
                field_name="family_c_shaping_ablation.shaping_defaults.lambda_default",
            ),
            max_total_shaping_magnitude_per_episode=_require_float(
                shaping_defaults["max_total_shaping_magnitude_per_episode"],
                field_name="family_c_shaping_ablation.shaping_defaults.max_total_shaping_magnitude_per_episode",
            ),
        ),
        truncation_reward=_require_float(
            body["truncation_reward"],
            field_name="family_c_shaping_ablation.truncation_reward",
        ),
    )


_COMPONENT_PARSERS = {
    "system": _parse_system_config,
    "model": _parse_model_config,
    "training_family_a": _parse_training_family_a_config,
    "environment": _parse_environment_config,
    "league": _parse_league_config,
    "evaluation": _parse_evaluation_config,
    "reproducibility": _parse_reproducibility_config,
    "metagame": _parse_metagame_config,
    "sensitivity": _parse_sensitivity_config,
    "compute_budget": _parse_compute_budget_config,
    "family_b_discount_ablation": _parse_family_b_discount_ablation_config,
    "family_c_shaping_ablation": _parse_family_c_shaping_ablation_config,
}


def load_stack_config(stack_path: Path | str) -> StackConfig:
    """Load the stack index, resolve component paths, and validate merged component dataclasses."""
    stack_file = Path(stack_path).resolve()
    root = stack_file.parents[1]
    doc = _load_yaml(stack_file)
    body = doc.get("rl_stack_locked", doc)
    if not isinstance(body, dict):
        raise ValueError("Missing `rl_stack_locked` mapping in stack config")

    raw_components = _require_mapping(body.get("components", {}), context="rl_stack_locked.components")
    raw_seed_sets = _require_mapping(body.get("seed_sets", {}), context="rl_stack_locked.seed_sets")
    components = {
        key: _resolve_repo_path(root, _require_text(value, field_name=f"rl_stack_locked.components.{key}"))
        for key, value in raw_components.items()
    }
    seed_sets = {
        key: _resolve_repo_path(root, _require_text(value, field_name=f"rl_stack_locked.seed_sets.{key}"))
        for key, value in raw_seed_sets.items()
    }

    component_docs: dict[str, dict[str, Any]] = {}
    parsed_components: dict[str, Any] = {}
    for component_name, component_path in components.items():
        if component_name not in _COMPONENT_PARSERS:
            raise ValueError(f"Unsupported component in stack config: {component_name}")
        component_doc = _load_component_doc(component_path, component_name)
        component_docs[component_name] = component_doc
        parsed_components[component_name] = _COMPONENT_PARSERS[component_name](component_doc)

    return StackConfig(
        root=root,
        schema_version=(
            _require_int(body["schema_version"], field_name="rl_stack_locked.schema_version", minimum=1)
            if "schema_version" in body
            else None
        ),
        description=_require_text(body["description"], field_name="rl_stack_locked.description") if "description" in body else "",
        lock_intent=_require_mapping(body.get("lock_intent", {}), context="rl_stack_locked.lock_intent"),
        components=components,
        seed_sets=seed_sets,
        component_docs=component_docs,
        config=LockedConfig(**parsed_components),
    )
