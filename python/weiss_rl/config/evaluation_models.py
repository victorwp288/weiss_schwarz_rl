"""Evaluation and reproducibility config records."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = [
    "DecisionKindTaggingConfig",
    "EvaluationConfig",
    "FinalPolicySetSelectionConfig",
    "FixedAnchorSetConfig",
    "IdsConfig",
    "LegalFingerprintChecksConfig",
    "LegalFingerprintConfig",
    "ReproducibilityConfig",
    "SeedDerivationConfig",
    "SpecBundlePolicyConfig",
    "StopRulesConfig",
]
