"""Strict YAML parsing and grouped preset loading for the RL config."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from weiss_rl.spec import normalize_spec_mismatch_policy, require_fail_on_spec_mismatch

from .compat import (
    resolve_legacy_config_path as _resolve_legacy_config_path,
)
from .compat import (
    resolve_repo_path as _resolve_repo_path,
)
from .compat import (
    resolve_repo_root as _resolve_repo_root,
)
from .models import (
    CurriculumCheckpointGuardConfig,
    CurriculumConfig,
    CurriculumEarlyCutoffConfig,
    CurriculumStallMonitorConfig,
    DecisionKindTaggingConfig,
    DeckSetSizeConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    FinalPolicySetSelectionConfig,
    FixedAnchorSetConfig,
    IdsConfig,
    LeagueConfig,
    LeaguePoolConfig,
    LeaguePromotionConfig,
    LeagueSamplingConfig,
    LeagueWarmupConfig,
    LegalFingerprintChecksConfig,
    LegalFingerprintConfig,
    LockedConfig,
    ModelConfig,
    ModelDropoutConfig,
    PromotionAnchorSetConfig,
    PromotionGateConfig,
    PromotionGateGuardrailsConfig,
    ReproducibilityConfig,
    RewardDiscountConfig,
    RewardsConfig,
    RewardShapingConfig,
    RewardTruncationConfig,
    SeedDerivationConfig,
    SpecBundlePolicyConfig,
    StackConfig,
    StopRulesConfig,
    SystemConfig,
    SystemProfileConfig,
)
from .training_parse import (
    _TRAINING_PUBLIC_HEURISTIC_PROFILES,
)
from .training_parse import (
    parse_training_config as _parse_training_config,
)
from .validation import (
    deep_merge as _deep_merge,
)
from .validation import (
    reject_unknown_keys as _reject_unknown_keys,
)
from .validation import (
    require_bool as _require_bool,
)
from .validation import (
    require_choice as _require_choice,
)
from .validation import (
    require_float as _require_float,
)
from .validation import (
    require_int as _require_int,
)
from .validation import (
    require_int_list as _require_int_list,
)
from .validation import (
    require_mapping as _require_mapping,
)
from .validation import (
    require_str_list as _require_str_list,
)
from .validation import (
    require_text as _require_text,
)

_EXPERIMENT_ROLES = frozenset(
    {
        "main",
        "thesis_multideck",
        "baseline_noleague",
        "baseline_noleague_multideck",
        "baseline_noleague_ablation_teacher_fade",
        "baseline_noleague_ablation_no_tactical_bias",
        "baseline_noleague_ablation_teacher_fade_no_tactical_bias",
        "baseline_norecurrence",
        "baseline_ppo_lite",
        "ablation_teacher_fade",
        "ablation_no_tactical_bias",
        "ablation_teacher_fade_no_tactical_bias",
        "ablation_no_b1_cutoff",
        "ablation_discount",
        "ablation_reward",
    }
)
_MODEL_ENCODER_KINDS = frozenset({"mlp", "typed_v1", "structured_v2"})
_STRUCTURED_POLICY_CONTRACTS = frozenset({"packed_v1", "factorized_v1"})
_MODEL_RECURRENT_CORES = frozenset({"gru", "none"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "extends",
        "description",
        "experiment",
        "system",
        "model",
        "training",
        "environment",
        "rewards",
        "curriculum",
        "league",
        "evaluation",
        "reproducibility",
        "seed_sets",
    }
)
_CANONICAL_CONFIG_KEYS = frozenset({"schema_version", "description", "config", "seed_sets"})
_CONFIG_SECTION_KEYS = frozenset(
    {
        "experiment",
        "system",
        "model",
        "training",
        "environment",
        "rewards",
        "curriculum",
        "league",
        "evaluation",
        "reproducibility",
    }
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _load_preset_document(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    active = set() if seen is None else seen
    if resolved in active:
        raise ValueError(f"Config extends cycle detected at {resolved}")
    active.add(resolved)
    doc = _load_yaml(resolved)
    _reject_unknown_keys(doc, allowed=_TOP_LEVEL_KEYS, context=str(resolved))
    merged: dict[str, Any] = {}
    parent = doc.get("extends")
    if parent is not None:
        parent_ref = _require_text(parent, field_name=f"{resolved}.extends")
        merged = _load_preset_document(
            _resolve_legacy_config_path((resolved.parent / parent_ref).resolve()), seen=active
        )
    active.remove(resolved)
    return _deep_merge(merged, doc)


def _parse_experiment_config(body: dict[str, Any]) -> ExperimentConfig:
    _reject_unknown_keys(body, allowed={"role"}, context="experiment")
    return ExperimentConfig(
        role=_require_choice(body["role"], field_name="experiment.role", allowed=_EXPERIMENT_ROLES),
    )


def _parse_system_config(body: dict[str, Any]) -> SystemConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "profile",
            "mp_start_method",
            "collection_backend",
            "learner_device",
            "actor_device",
            "actor_process_count",
            "envs_per_actor",
            "total_envs",
            "actor_torch_threads",
            "learner_torch_threads",
            "actor_queue_capacity_unrolls",
            "learner_prefetch_batches",
        },
        context="system",
    )
    profile = _require_mapping(body["profile"], context="system.profile")
    _reject_unknown_keys(
        profile,
        allowed={"training", "local_iteration", "ci_invariant_testing"},
        context="system.profile",
    )
    return SystemConfig(
        profile=SystemProfileConfig(
            training=_require_text(profile["training"], field_name="system.profile.training"),
            local_iteration=_require_text(profile["local_iteration"], field_name="system.profile.local_iteration"),
            ci_invariant_testing=_require_text(
                profile["ci_invariant_testing"],
                field_name="system.profile.ci_invariant_testing",
            ),
        ),
        mp_start_method=_require_text(body["mp_start_method"], field_name="system.mp_start_method"),
        collection_backend=_require_choice(
            body.get("collection_backend", "auto"),
            field_name="system.collection_backend",
            allowed=("auto", "central", "process"),
        ),
        learner_device=_require_text(body["learner_device"], field_name="system.learner_device"),
        actor_device=_require_text(body["actor_device"], field_name="system.actor_device"),
        actor_process_count=_require_int(
            body["actor_process_count"], field_name="system.actor_process_count", minimum=1
        ),
        envs_per_actor=_require_int(body["envs_per_actor"], field_name="system.envs_per_actor", minimum=1),
        total_envs=_require_int(body["total_envs"], field_name="system.total_envs", minimum=1),
        actor_torch_threads=_require_int(
            body["actor_torch_threads"], field_name="system.actor_torch_threads", minimum=1
        ),
        learner_torch_threads=_require_int(
            body["learner_torch_threads"], field_name="system.learner_torch_threads", minimum=1
        ),
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
    _reject_unknown_keys(
        body,
        allowed={
            "gru_hidden_size",
            "encoder_mlp_width",
            "encoder_mlp_layers",
            "encoder_kind",
            "structured_policy_contract",
            "typed_feature_width",
            "recurrent_core",
            "candidate_scoring_chunk_size",
            "cuda_learner_candidate_scoring_chunk_size",
            "public_heuristic_logit_bias_scale",
            "public_heuristic_actor_logit_bias_scale",
            "public_heuristic_logit_bias_start_updates",
            "public_heuristic_logit_bias_end_updates",
            "public_heuristic_logit_bias_final_scale",
            "public_heuristic_logit_bias_families",
            "public_heuristic_logit_bias_profile",
            "layer_norm",
            "dropout",
        },
        context="model",
    )
    dropout = _require_mapping(body["dropout"], context="model.dropout")
    _reject_unknown_keys(dropout, allowed={"family_a", "ablation"}, context="model.dropout")
    public_heuristic_logit_bias_start_updates = _require_int(
        body.get("public_heuristic_logit_bias_start_updates", 0),
        field_name="model.public_heuristic_logit_bias_start_updates",
        minimum=0,
    )
    public_heuristic_logit_bias_end_updates = _require_int(
        body.get("public_heuristic_logit_bias_end_updates", -1),
        field_name="model.public_heuristic_logit_bias_end_updates",
        minimum=-1,
    )
    if (
        public_heuristic_logit_bias_end_updates >= 0
        and public_heuristic_logit_bias_end_updates < public_heuristic_logit_bias_start_updates
    ):
        raise ValueError(
            "model.public_heuristic_logit_bias_end_updates must be >= model.public_heuristic_logit_bias_start_updates"
        )
    public_heuristic_logit_bias_final_scale = _require_float(
        body.get(
            "public_heuristic_logit_bias_final_scale",
            body.get("public_heuristic_logit_bias_scale", 0.0),
        ),
        field_name="model.public_heuristic_logit_bias_final_scale",
    )
    if public_heuristic_logit_bias_final_scale < 0.0:
        raise ValueError("model.public_heuristic_logit_bias_final_scale must be >= 0.0")
    return ModelConfig(
        gru_hidden_size=_require_int(body["gru_hidden_size"], field_name="model.gru_hidden_size", minimum=1),
        encoder_mlp_width=_require_int(body["encoder_mlp_width"], field_name="model.encoder_mlp_width", minimum=1),
        encoder_mlp_layers=_require_int(body["encoder_mlp_layers"], field_name="model.encoder_mlp_layers", minimum=1),
        encoder_kind=_require_choice(
            body.get("encoder_kind", "mlp"),
            field_name="model.encoder_kind",
            allowed=_MODEL_ENCODER_KINDS,
        ),
        structured_policy_contract=_require_choice(
            body.get("structured_policy_contract", "packed_v1"),
            field_name="model.structured_policy_contract",
            allowed=_STRUCTURED_POLICY_CONTRACTS,
        ),
        typed_feature_width=_require_int(
            body.get("typed_feature_width", 64),
            field_name="model.typed_feature_width",
            minimum=1,
        ),
        recurrent_core=_require_choice(
            body.get("recurrent_core", "gru"),
            field_name="model.recurrent_core",
            allowed=_MODEL_RECURRENT_CORES,
        ),
        candidate_scoring_chunk_size=_require_int(
            body.get("candidate_scoring_chunk_size", 65536),
            field_name="model.candidate_scoring_chunk_size",
            minimum=1,
        ),
        cuda_learner_candidate_scoring_chunk_size=_require_int(
            body.get("cuda_learner_candidate_scoring_chunk_size", 262144),
            field_name="model.cuda_learner_candidate_scoring_chunk_size",
            minimum=1,
        ),
        public_heuristic_logit_bias_scale=_require_float(
            body.get("public_heuristic_logit_bias_scale", 0.0),
            field_name="model.public_heuristic_logit_bias_scale",
        ),
        public_heuristic_actor_logit_bias_scale=_require_float(
            body.get("public_heuristic_actor_logit_bias_scale", -1.0),
            field_name="model.public_heuristic_actor_logit_bias_scale",
        ),
        public_heuristic_logit_bias_start_updates=public_heuristic_logit_bias_start_updates,
        public_heuristic_logit_bias_end_updates=public_heuristic_logit_bias_end_updates,
        public_heuristic_logit_bias_final_scale=_require_float(
            public_heuristic_logit_bias_final_scale,
            field_name="model.public_heuristic_logit_bias_final_scale",
        ),
        public_heuristic_logit_bias_families=_require_str_list(
            body.get("public_heuristic_logit_bias_families", []),
            field_name="model.public_heuristic_logit_bias_families",
        ),
        public_heuristic_logit_bias_profile=_require_choice(
            body.get("public_heuristic_logit_bias_profile", "base"),
            field_name="model.public_heuristic_logit_bias_profile",
            allowed=_TRAINING_PUBLIC_HEURISTIC_PROFILES,
        ),
        layer_norm=_require_bool(body["layer_norm"], field_name="model.layer_norm"),
        dropout=ModelDropoutConfig(
            family_a=_require_float(dropout["family_a"], field_name="model.dropout.family_a"),
            ablation=_require_float(dropout["ablation"], field_name="model.dropout.ablation"),
        ),
    )


def _parse_environment_config(body: dict[str, Any]) -> EnvironmentConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "observation_visibility",
            "visibility",
            "truncate_on_max_steps",
            "max_raw_decisions_per_episode",
            "max_decisions",
            "max_decisions_per_episode",
            "max_learner_steps_per_episode",
            "max_ticks",
            "deck_set_size",
            "deck_pool",
            "opponent_deck_pool",
        },
        context="environment",
    )
    deck_set_size = _require_mapping(body["deck_set_size"], context="environment.deck_set_size")
    _reject_unknown_keys(deck_set_size, allowed={"bring_up", "paper"}, context="environment.deck_set_size")
    return EnvironmentConfig(
        observation_visibility=_require_text(
            body["observation_visibility"],
            field_name="environment.observation_visibility",
        ),
        visibility=_require_text(body["visibility"], field_name="environment.visibility"),
        truncate_on_max_steps=_require_bool(
            body["truncate_on_max_steps"], field_name="environment.truncate_on_max_steps"
        ),
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
        deck_set_size=DeckSetSizeConfig(
            bring_up=_require_int(
                deck_set_size["bring_up"], field_name="environment.deck_set_size.bring_up", minimum=1
            ),
            paper=_require_int(deck_set_size["paper"], field_name="environment.deck_set_size.paper", minimum=1),
        ),
        deck_pool=_require_str_list(body.get("deck_pool", []), field_name="environment.deck_pool"),
        opponent_deck_pool=_require_str_list(
            body.get("opponent_deck_pool", []),
            field_name="environment.opponent_deck_pool",
        ),
    )


def _parse_rewards_config(body: dict[str, Any]) -> RewardsConfig:
    _reject_unknown_keys(body, allowed={"objective", "discount", "shaping", "truncation"}, context="rewards")
    discount = _require_mapping(body["discount"], context="rewards.discount")
    shaping = _require_mapping(body["shaping"], context="rewards.shaping")
    truncation = _require_mapping(body["truncation"], context="rewards.truncation")
    _reject_unknown_keys(discount, allowed={"gamma"}, context="rewards.discount")
    _reject_unknown_keys(
        shaping,
        allowed={
            "enable_damage_shaping",
            "damage_reward",
            "level_reward",
            "board_reward",
            "no_progress_penalty",
            "pass_with_nonpass_penalty",
        },
        context="rewards.shaping",
    )
    _reject_unknown_keys(
        truncation,
        allowed={"reward", "bootstrap_value", "bootstrap_rule"},
        context="rewards.truncation",
    )
    return RewardsConfig(
        objective=_require_text(body["objective"], field_name="rewards.objective"),
        discount=RewardDiscountConfig(
            gamma=_require_float(discount["gamma"], field_name="rewards.discount.gamma"),
        ),
        shaping=RewardShapingConfig(
            enable_damage_shaping=_require_bool(
                shaping["enable_damage_shaping"],
                field_name="rewards.shaping.enable_damage_shaping",
            ),
            damage_reward=_require_float(shaping["damage_reward"], field_name="rewards.shaping.damage_reward"),
            level_reward=_require_float(shaping.get("level_reward", 0.0), field_name="rewards.shaping.level_reward"),
            board_reward=_require_float(shaping.get("board_reward", 0.0), field_name="rewards.shaping.board_reward"),
            no_progress_penalty=_require_float(
                shaping.get("no_progress_penalty", 0.0),
                field_name="rewards.shaping.no_progress_penalty",
            ),
            pass_with_nonpass_penalty=_require_float(
                shaping.get("pass_with_nonpass_penalty", 0.0),
                field_name="rewards.shaping.pass_with_nonpass_penalty",
            ),
        ),
        truncation=RewardTruncationConfig(
            reward=_require_float(truncation["reward"], field_name="rewards.truncation.reward"),
            bootstrap_value=_require_bool(
                truncation["bootstrap_value"],
                field_name="rewards.truncation.bootstrap_value",
            ),
            bootstrap_rule=_require_text(
                truncation["bootstrap_rule"],
                field_name="rewards.truncation.bootstrap_rule",
            ),
        ),
    )


def _normalize_curriculum_payload(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_curriculum_payload(item, field_name=f"{field_name}[]") for item in value]
    if isinstance(value, Mapping):
        return {
            _require_text(key, field_name=f"{field_name}.<key>"): _normalize_curriculum_payload(
                item,
                field_name=f"{field_name}.{key}",
            )
            for key, item in value.items()
        }
    raise ValueError(f"{field_name} contains unsupported value type: {type(value).__name__}")


def _parse_curriculum_config(body: dict[str, Any] | None) -> CurriculumConfig:
    if body is None:
        return CurriculumConfig()
    _reject_unknown_keys(
        body,
        allowed={"simulator", "stall_monitor", "early_cutoff", "checkpoint_guard"},
        context="curriculum",
    )
    simulator = _require_mapping(body.get("simulator", {}), context="curriculum.simulator")
    stall_monitor = _require_mapping(body.get("stall_monitor", {}), context="curriculum.stall_monitor")
    early_cutoff = _require_mapping(body.get("early_cutoff", {}), context="curriculum.early_cutoff")
    checkpoint_guard = _require_mapping(body.get("checkpoint_guard", {}), context="curriculum.checkpoint_guard")
    _reject_unknown_keys(
        stall_monitor,
        allowed={"enabled", "truncation_rate_threshold", "consecutive_evals"},
        context="curriculum.stall_monitor",
    )
    _reject_unknown_keys(
        early_cutoff,
        allowed={
            "enabled",
            "warmup_updates",
            "patience_updates",
            "min_improvement",
            "stall_patience_evals",
            "stall_rate_threshold",
        },
        context="curriculum.early_cutoff",
    )
    _reject_unknown_keys(
        checkpoint_guard,
        allowed={
            "enabled",
            "rollback_score_margin",
            "rollback_truncation_rate_threshold",
            "rollback_max_prob_lt_half",
            "min_best_score",
            "promote_min_prob_gt_half",
            "promote_max_ci_half_width",
            "cooldown_updates",
        },
        context="curriculum.checkpoint_guard",
    )
    return CurriculumConfig(
        simulator={
            key: _normalize_curriculum_payload(value, field_name=f"curriculum.simulator.{key}")
            for key, value in simulator.items()
        },
        stall_monitor=CurriculumStallMonitorConfig(
            enabled=_require_bool(stall_monitor.get("enabled", False), field_name="curriculum.stall_monitor.enabled"),
            truncation_rate_threshold=_require_float(
                stall_monitor.get("truncation_rate_threshold", 1.0),
                field_name="curriculum.stall_monitor.truncation_rate_threshold",
            ),
            consecutive_evals=_require_int(
                stall_monitor.get("consecutive_evals", 2),
                field_name="curriculum.stall_monitor.consecutive_evals",
                minimum=1,
            ),
        ),
        early_cutoff=CurriculumEarlyCutoffConfig(
            enabled=_require_bool(early_cutoff.get("enabled", False), field_name="curriculum.early_cutoff.enabled"),
            warmup_updates=_require_int(
                early_cutoff.get("warmup_updates", 0),
                field_name="curriculum.early_cutoff.warmup_updates",
                minimum=0,
            ),
            patience_updates=_require_int(
                early_cutoff.get("patience_updates", 0),
                field_name="curriculum.early_cutoff.patience_updates",
                minimum=0,
            ),
            min_improvement=_require_float(
                early_cutoff.get("min_improvement", 0.0),
                field_name="curriculum.early_cutoff.min_improvement",
            ),
            stall_patience_evals=_require_int(
                early_cutoff.get("stall_patience_evals", 0),
                field_name="curriculum.early_cutoff.stall_patience_evals",
                minimum=0,
            ),
            stall_rate_threshold=_require_float(
                early_cutoff.get("stall_rate_threshold", 1.0),
                field_name="curriculum.early_cutoff.stall_rate_threshold",
            ),
        ),
        checkpoint_guard=CurriculumCheckpointGuardConfig(
            enabled=_require_bool(
                checkpoint_guard.get("enabled", False),
                field_name="curriculum.checkpoint_guard.enabled",
            ),
            rollback_score_margin=_require_float(
                checkpoint_guard.get("rollback_score_margin", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_score_margin",
            ),
            rollback_truncation_rate_threshold=_require_float(
                checkpoint_guard.get("rollback_truncation_rate_threshold", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_truncation_rate_threshold",
            ),
            rollback_max_prob_lt_half=_require_float(
                checkpoint_guard.get("rollback_max_prob_lt_half", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_max_prob_lt_half",
            ),
            min_best_score=_require_float(
                checkpoint_guard.get("min_best_score", 1.0),
                field_name="curriculum.checkpoint_guard.min_best_score",
            ),
            promote_min_prob_gt_half=_require_float(
                checkpoint_guard.get("promote_min_prob_gt_half", 0.0),
                field_name="curriculum.checkpoint_guard.promote_min_prob_gt_half",
            ),
            promote_max_ci_half_width=_require_float(
                checkpoint_guard.get("promote_max_ci_half_width", 1.0),
                field_name="curriculum.checkpoint_guard.promote_max_ci_half_width",
            ),
            cooldown_updates=_require_int(
                checkpoint_guard.get("cooldown_updates", 0),
                field_name="curriculum.checkpoint_guard.cooldown_updates",
                minimum=0,
            ),
        ),
    )


def _parse_league_config(body: dict[str, Any]) -> LeagueConfig:
    _reject_unknown_keys(body, allowed={"enabled", "pool", "sampling", "warmup", "promotion"}, context="league")
    pool = _require_mapping(body["pool"], context="league.pool")
    sampling = _require_mapping(body["sampling"], context="league.sampling")
    warmup = _require_mapping(body["warmup"], context="league.warmup")
    promotion = _require_mapping(body["promotion"], context="league.promotion")
    anchor_set = _require_mapping(promotion["anchor_set_v1"], context="league.promotion.anchor_set_v1")
    gate = _require_mapping(promotion["gate"], context="league.promotion.gate")
    guardrails = _require_mapping(gate["guardrails"], context="league.promotion.gate.guardrails")

    _reject_unknown_keys(
        pool,
        allowed={"recent_size", "champion_size", "champion_max_age_updates"},
        context="league.pool",
    )
    _reject_unknown_keys(
        sampling,
        allowed={
            "opponent_sampling",
            "pfsp_power",
            "pfsp_epsilon_uniform",
            "pfsp_stats_source",
            "pfsp_window_episodes",
            "heuristic_public_start_updates",
            "heuristic_public_mix_fraction",
            "heuristic_public_mix_end_updates",
            "heuristic_public_final_mix_fraction",
            "heuristic_public_variant_mix_fraction",
            "heuristic_public_variant_mix_end_updates",
            "heuristic_public_variant_final_mix_fraction",
            "noleague_baseline_mix_fraction",
            "noleague_baseline_mix_end_updates",
            "noleague_baseline_reward_scale",
            "noleague_baseline_force_focal_seat",
            "warmup_snapshot_mix_fraction",
            "exclude_seed_snapshots_from_pfsp",
            "mirror_mix_fraction",
            "heuristic_public_reserved_envs_per_actor",
            "noleague_baseline_reserved_envs_per_actor",
            "champion_mix_fraction",
            "hard_negative_mix_fraction",
            "hard_negative_min_samples",
            "hard_negative_max_win_rate",
        },
        context="league.sampling",
    )
    _reject_unknown_keys(
        warmup,
        allowed={
            "first_updates",
            "initial_window_episodes",
            "ramp_target_updates",
            "ramp_target_window_episodes",
            "eval_gate_enabled",
            "eval_gate_min_anchor_scores",
            "eval_gate_min_aggregate_score",
        },
        context="league.warmup",
    )
    _reject_unknown_keys(
        promotion,
        allowed={"enabled", "paired_seeds", "threshold", "anchor_set_v1", "seed_file", "gate"},
        context="league.promotion",
    )
    _reject_unknown_keys(
        anchor_set, allowed={"required", "optional_if_available"}, context="league.promotion.anchor_set_v1"
    )
    _reject_unknown_keys(
        gate,
        allowed={
            "uncertainty_method",
            "weighting",
            "seat_swap",
            "folding",
            "guardrails",
            "record_file",
            "target_min_anchor_scores",
            "async_enabled",
            "parallel_workers",
            "parallel_worker_devices",
        },
        context="league.promotion.gate",
    )
    _reject_unknown_keys(
        guardrails,
        allowed={"max_prob_anchor_loss_below_0_45", "max_truncation_rate"},
        context="league.promotion.gate.guardrails",
    )

    pfsp_stats_source = _require_text(
        sampling["pfsp_stats_source"],
        field_name="league.sampling.pfsp_stats_source",
    )
    if pfsp_stats_source != "online_outcomes":
        raise ValueError("league.sampling.pfsp_stats_source currently only supports 'online_outcomes'")
    eval_gate_min_anchor_scores_raw = warmup.get("eval_gate_min_anchor_scores", {})
    if not isinstance(eval_gate_min_anchor_scores_raw, Mapping):
        raise TypeError("league.warmup.eval_gate_min_anchor_scores must be a mapping")
    eval_gate_min_anchor_scores: dict[str, float] = {}
    for anchor_name, score in eval_gate_min_anchor_scores_raw.items():
        anchor_name_text = str(anchor_name).strip()
        if not anchor_name_text:
            raise ValueError("league.warmup.eval_gate_min_anchor_scores anchor names must be non-empty")
        score_f = float(score)
        if not math.isfinite(score_f) or score_f < 0.0 or score_f > 1.0:
            raise ValueError("league.warmup.eval_gate_min_anchor_scores values must be finite probabilities in [0, 1]")
        eval_gate_min_anchor_scores[anchor_name_text] = score_f
    eval_gate_min_aggregate_score_raw = warmup.get("eval_gate_min_aggregate_score", None)
    eval_gate_min_aggregate_score = None
    if eval_gate_min_aggregate_score_raw is not None:
        eval_gate_min_aggregate_score = float(eval_gate_min_aggregate_score_raw)
        if (
            not math.isfinite(eval_gate_min_aggregate_score)
            or eval_gate_min_aggregate_score < 0.0
            or eval_gate_min_aggregate_score > 1.0
        ):
            raise ValueError("league.warmup.eval_gate_min_aggregate_score must be a finite probability in [0, 1]")
    target_min_anchor_scores_raw = gate.get("target_min_anchor_scores", {})
    if not isinstance(target_min_anchor_scores_raw, Mapping):
        raise TypeError("league.promotion.gate.target_min_anchor_scores must be a mapping")
    target_min_anchor_scores: dict[str, float] = {}
    for anchor_name, score in target_min_anchor_scores_raw.items():
        anchor_name_text = str(anchor_name).strip()
        if not anchor_name_text:
            raise ValueError("league.promotion.gate.target_min_anchor_scores anchor names must be non-empty")
        score_f = float(score)
        if not math.isfinite(score_f) or score_f < 0.0 or score_f > 1.0:
            raise ValueError(
                "league.promotion.gate.target_min_anchor_scores values must be finite probabilities in [0, 1]"
            )
        target_min_anchor_scores[anchor_name_text] = score_f

    return LeagueConfig(
        enabled=_require_bool(body["enabled"], field_name="league.enabled"),
        pool=LeaguePoolConfig(
            recent_size=_require_int(pool["recent_size"], field_name="league.pool.recent_size", minimum=1),
            champion_size=_require_int(pool["champion_size"], field_name="league.pool.champion_size", minimum=0),
            champion_max_age_updates=_require_int(
                pool.get("champion_max_age_updates", 0),
                field_name="league.pool.champion_max_age_updates",
                minimum=0,
            ),
        ),
        sampling=LeagueSamplingConfig(
            opponent_sampling=_require_text(
                sampling["opponent_sampling"], field_name="league.sampling.opponent_sampling"
            ),
            pfsp_power=_require_float(sampling["pfsp_power"], field_name="league.sampling.pfsp_power"),
            pfsp_epsilon_uniform=_require_float(
                sampling["pfsp_epsilon_uniform"],
                field_name="league.sampling.pfsp_epsilon_uniform",
            ),
            pfsp_stats_source=pfsp_stats_source,
            pfsp_window_episodes=_require_int(
                sampling["pfsp_window_episodes"],
                field_name="league.sampling.pfsp_window_episodes",
                minimum=1,
            ),
            heuristic_public_start_updates=_require_int(
                sampling.get("heuristic_public_start_updates", 0),
                field_name="league.sampling.heuristic_public_start_updates",
                minimum=0,
            ),
            heuristic_public_mix_fraction=_require_float(
                sampling.get("heuristic_public_mix_fraction", 0.0),
                field_name="league.sampling.heuristic_public_mix_fraction",
            ),
            heuristic_public_mix_end_updates=_require_int(
                sampling.get("heuristic_public_mix_end_updates", -1),
                field_name="league.sampling.heuristic_public_mix_end_updates",
                minimum=-1,
            ),
            heuristic_public_final_mix_fraction=_require_float(
                sampling.get(
                    "heuristic_public_final_mix_fraction",
                    sampling.get("heuristic_public_mix_fraction", 0.0),
                ),
                field_name="league.sampling.heuristic_public_final_mix_fraction",
            ),
            heuristic_public_variant_mix_fraction=_require_float(
                sampling.get("heuristic_public_variant_mix_fraction", 0.0),
                field_name="league.sampling.heuristic_public_variant_mix_fraction",
            ),
            heuristic_public_variant_mix_end_updates=_require_int(
                sampling.get("heuristic_public_variant_mix_end_updates", -1),
                field_name="league.sampling.heuristic_public_variant_mix_end_updates",
                minimum=-1,
            ),
            heuristic_public_variant_final_mix_fraction=_require_float(
                sampling.get(
                    "heuristic_public_variant_final_mix_fraction",
                    sampling.get("heuristic_public_variant_mix_fraction", 0.0),
                ),
                field_name="league.sampling.heuristic_public_variant_final_mix_fraction",
            ),
            noleague_baseline_mix_fraction=_require_float(
                sampling.get("noleague_baseline_mix_fraction", 0.0),
                field_name="league.sampling.noleague_baseline_mix_fraction",
            ),
            noleague_baseline_mix_end_updates=_require_int(
                sampling.get("noleague_baseline_mix_end_updates", -1),
                field_name="league.sampling.noleague_baseline_mix_end_updates",
                minimum=-1,
            ),
            noleague_baseline_reward_scale=_require_float(
                sampling.get("noleague_baseline_reward_scale", 1.0),
                field_name="league.sampling.noleague_baseline_reward_scale",
            ),
            noleague_baseline_force_focal_seat=int(
                _require_choice(
                    str(
                        _require_int(
                            sampling.get("noleague_baseline_force_focal_seat", -1),
                            field_name="league.sampling.noleague_baseline_force_focal_seat",
                            minimum=-1,
                        )
                    ),
                    allowed={"-1", "0", "1"},
                    field_name="league.sampling.noleague_baseline_force_focal_seat",
                )
            ),
            warmup_snapshot_mix_fraction=_require_float(
                sampling.get("warmup_snapshot_mix_fraction", 0.0),
                field_name="league.sampling.warmup_snapshot_mix_fraction",
            ),
            exclude_seed_snapshots_from_pfsp=_require_bool(
                sampling.get("exclude_seed_snapshots_from_pfsp", False),
                field_name="league.sampling.exclude_seed_snapshots_from_pfsp",
            ),
            mirror_mix_fraction=_require_float(
                sampling.get("mirror_mix_fraction", 0.0),
                field_name="league.sampling.mirror_mix_fraction",
            ),
            heuristic_public_reserved_envs_per_actor=_require_int(
                sampling.get("heuristic_public_reserved_envs_per_actor", 0),
                field_name="league.sampling.heuristic_public_reserved_envs_per_actor",
                minimum=0,
            ),
            noleague_baseline_reserved_envs_per_actor=_require_int(
                sampling.get("noleague_baseline_reserved_envs_per_actor", 0),
                field_name="league.sampling.noleague_baseline_reserved_envs_per_actor",
                minimum=0,
            ),
            champion_mix_fraction=_require_float(
                sampling.get("champion_mix_fraction", 0.35),
                field_name="league.sampling.champion_mix_fraction",
            ),
            hard_negative_mix_fraction=_require_float(
                sampling.get("hard_negative_mix_fraction", 0.2),
                field_name="league.sampling.hard_negative_mix_fraction",
            ),
            hard_negative_min_samples=_require_int(
                sampling.get("hard_negative_min_samples", 16),
                field_name="league.sampling.hard_negative_min_samples",
                minimum=1,
            ),
            hard_negative_max_win_rate=_require_float(
                sampling.get("hard_negative_max_win_rate", 0.45),
                field_name="league.sampling.hard_negative_max_win_rate",
            ),
        ),
        warmup=LeagueWarmupConfig(
            first_updates=_require_int(warmup["first_updates"], field_name="league.warmup.first_updates", minimum=0),
            initial_window_episodes=_require_int(
                warmup["initial_window_episodes"],
                field_name="league.warmup.initial_window_episodes",
                minimum=0,
            ),
            ramp_target_updates=_require_int(
                warmup["ramp_target_updates"],
                field_name="league.warmup.ramp_target_updates",
                minimum=0,
            ),
            ramp_target_window_episodes=_require_int(
                warmup["ramp_target_window_episodes"],
                field_name="league.warmup.ramp_target_window_episodes",
                minimum=0,
            ),
            eval_gate_enabled=_require_bool(
                warmup.get("eval_gate_enabled", False),
                field_name="league.warmup.eval_gate_enabled",
            ),
            eval_gate_min_anchor_scores=eval_gate_min_anchor_scores,
            eval_gate_min_aggregate_score=eval_gate_min_aggregate_score,
        ),
        promotion=LeaguePromotionConfig(
            enabled=_require_bool(promotion["enabled"], field_name="league.promotion.enabled"),
            paired_seeds=_require_int(promotion["paired_seeds"], field_name="league.promotion.paired_seeds", minimum=1),
            threshold=_require_text(promotion["threshold"], field_name="league.promotion.threshold"),
            anchor_set_v1=PromotionAnchorSetConfig(
                required=_require_str_list(
                    anchor_set["required"], field_name="league.promotion.anchor_set_v1.required"
                ),
                optional_if_available=_require_str_list(
                    anchor_set["optional_if_available"],
                    field_name="league.promotion.anchor_set_v1.optional_if_available",
                ),
            ),
            seed_file=_require_text(promotion["seed_file"], field_name="league.promotion.seed_file"),
            gate=PromotionGateConfig(
                uncertainty_method=_require_text(
                    gate["uncertainty_method"],
                    field_name="league.promotion.gate.uncertainty_method",
                ),
                weighting=_require_text(gate["weighting"], field_name="league.promotion.gate.weighting"),
                seat_swap=_require_bool(gate["seat_swap"], field_name="league.promotion.gate.seat_swap"),
                folding=_require_text(gate["folding"], field_name="league.promotion.gate.folding"),
                guardrails=PromotionGateGuardrailsConfig(
                    max_prob_anchor_loss_below_0_45=_require_float(
                        guardrails["max_prob_anchor_loss_below_0_45"],
                        field_name="league.promotion.gate.guardrails.max_prob_anchor_loss_below_0_45",
                    ),
                    max_truncation_rate=_require_float(
                        guardrails["max_truncation_rate"],
                        field_name="league.promotion.gate.guardrails.max_truncation_rate",
                    ),
                ),
                record_file=_require_text(gate["record_file"], field_name="league.promotion.gate.record_file"),
                target_min_anchor_scores=target_min_anchor_scores,
                async_enabled=_require_bool(
                    gate.get("async_enabled", False),
                    field_name="league.promotion.gate.async_enabled",
                ),
                parallel_workers=_require_int(
                    gate.get("parallel_workers", 1),
                    field_name="league.promotion.gate.parallel_workers",
                    minimum=1,
                ),
                parallel_worker_devices=_require_str_list(
                    gate.get("parallel_worker_devices", []),
                    field_name="league.promotion.gate.parallel_worker_devices",
                ),
            ),
        ),
    )


def _parse_evaluation_config(body: dict[str, Any]) -> EvaluationConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "seat_swap",
            "eval_device",
            "async_periodic_dev_eval_enabled",
            "periodic_dev_eval_parallel_workers",
            "periodic_dev_eval_parallel_worker_devices",
            "periodic_dev_eval_batched_inference_enabled",
            "eval_inference_mode",
            "eval_sampling_algorithm",
            "eval_assert_sorted_legal_ids",
            "seed_files",
            "periodic_dev_eval_interval_updates",
            "periodic_dev_eval_paired_seeds",
            "periodic_dev_eval_anchor_weights",
            "final_policy_set_size",
            "final_matrix_stage1_paired_seeds",
            "final_matrix_stage2_adaptive_max_paired_seeds",
            "stop_rules",
            "replay_capture_rate_eval",
            "regression_capture_count",
            "legal_fingerprint_checks",
            "decision_kind_tagging",
            "final_policy_set_selection",
        },
        context="evaluation",
    )
    seed_files = _require_mapping(body["seed_files"], context="evaluation.seed_files")
    stop_rules = _require_mapping(body["stop_rules"], context="evaluation.stop_rules")
    legal = _require_mapping(body["legal_fingerprint_checks"], context="evaluation.legal_fingerprint_checks")
    decision = _require_mapping(body["decision_kind_tagging"], context="evaluation.decision_kind_tagging")
    selection = _require_mapping(body["final_policy_set_selection"], context="evaluation.final_policy_set_selection")
    fixed_anchor = _require_mapping(
        selection["fixed_anchor_set_v1"], context="evaluation.final_policy_set_selection.fixed_anchor_set_v1"
    )
    periodic_dev_eval_anchor_weights_raw = _require_mapping(
        body.get("periodic_dev_eval_anchor_weights", {}),
        context="evaluation.periodic_dev_eval_anchor_weights",
    )
    periodic_dev_eval_anchor_weights = {
        str(key).strip(): _require_float(
            value,
            field_name=f"evaluation.periodic_dev_eval_anchor_weights.{key}",
        )
        for key, value in periodic_dev_eval_anchor_weights_raw.items()
        if str(key).strip()
    }
    invalid_anchor_weights = {
        key: value for key, value in periodic_dev_eval_anchor_weights.items() if not math.isfinite(value) or value < 0.0
    }
    if invalid_anchor_weights:
        raise ValueError("evaluation.periodic_dev_eval_anchor_weights values must be finite and >= 0.0")
    _reject_unknown_keys(
        stop_rules, allowed={"stop_delta_ci_half_width", "stop_confidence"}, context="evaluation.stop_rules"
    )
    _reject_unknown_keys(
        legal,
        allowed={"enabled", "version", "require_strictly_increasing_legal_ids", "mismatch_policy"},
        context="evaluation.legal_fingerprint_checks",
    )
    _reject_unknown_keys(
        decision,
        allowed={"required_for_training", "enable_python_derived_debug_tag"},
        context="evaluation.decision_kind_tagging",
    )
    _reject_unknown_keys(
        selection,
        allowed={
            "version",
            "include_random_legal_baseline_b0",
            "include_no_league_baseline_b1",
            "include_heuristic_public_b2_if_exists",
            "include_final_champion_snapshot",
            "include_spaced_snapshots_near_percent_updates",
            "remaining_slots_strategy",
            "fixed_anchor_set_v1",
            "seed_file",
            "folding",
            "seat_swap",
            "tie_break",
        },
        context="evaluation.final_policy_set_selection",
    )
    _reject_unknown_keys(
        fixed_anchor,
        allowed={"required", "optional_if_available"},
        context="evaluation.final_policy_set_selection.fixed_anchor_set_v1",
    )
    mismatch_policy = _require_text(
        legal["mismatch_policy"],
        field_name="evaluation.legal_fingerprint_checks.mismatch_policy",
    )
    if mismatch_policy != "hard_fail":
        raise ValueError(
            f"evaluation.legal_fingerprint_checks.mismatch_policy must be 'hard_fail', got {mismatch_policy!r}"
        )
    return EvaluationConfig(
        seat_swap=_require_bool(body["seat_swap"], field_name="evaluation.seat_swap"),
        eval_device=_require_text(body["eval_device"], field_name="evaluation.eval_device"),
        async_periodic_dev_eval_enabled=_require_bool(
            body.get("async_periodic_dev_eval_enabled", False),
            field_name="evaluation.async_periodic_dev_eval_enabled",
        ),
        periodic_dev_eval_parallel_workers=_require_int(
            body.get("periodic_dev_eval_parallel_workers", 1),
            field_name="evaluation.periodic_dev_eval_parallel_workers",
            minimum=1,
        ),
        periodic_dev_eval_parallel_worker_devices=_require_str_list(
            body.get("periodic_dev_eval_parallel_worker_devices", []),
            field_name="evaluation.periodic_dev_eval_parallel_worker_devices",
        ),
        periodic_dev_eval_batched_inference_enabled=_require_bool(
            body.get("periodic_dev_eval_batched_inference_enabled", False),
            field_name="evaluation.periodic_dev_eval_batched_inference_enabled",
        ),
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
            key: _require_text(value, field_name=f"evaluation.seed_files.{key}") for key, value in seed_files.items()
        },
        periodic_dev_eval_interval_updates=_require_int(
            body["periodic_dev_eval_interval_updates"],
            field_name="evaluation.periodic_dev_eval_interval_updates",
            minimum=0,
        ),
        periodic_dev_eval_paired_seeds=_require_int(
            body["periodic_dev_eval_paired_seeds"],
            field_name="evaluation.periodic_dev_eval_paired_seeds",
            minimum=1,
        ),
        periodic_dev_eval_anchor_weights=periodic_dev_eval_anchor_weights,
        final_policy_set_size=_require_int(
            body["final_policy_set_size"], field_name="evaluation.final_policy_set_size", minimum=1
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
            minimum=0,
        ),
        legal_fingerprint_checks=LegalFingerprintChecksConfig(
            enabled=_require_bool(legal["enabled"], field_name="evaluation.legal_fingerprint_checks.enabled"),
            version=_require_text(legal["version"], field_name="evaluation.legal_fingerprint_checks.version"),
            require_strictly_increasing_legal_ids=_require_bool(
                legal["require_strictly_increasing_legal_ids"],
                field_name="evaluation.legal_fingerprint_checks.require_strictly_increasing_legal_ids",
            ),
            mismatch_policy=_require_text(
                mismatch_policy,
                field_name="evaluation.legal_fingerprint_checks.mismatch_policy",
            ),
        ),
        decision_kind_tagging=DecisionKindTaggingConfig(
            required_for_training=_require_bool(
                decision["required_for_training"],
                field_name="evaluation.decision_kind_tagging.required_for_training",
            ),
            enable_python_derived_debug_tag=_require_bool(
                decision["enable_python_derived_debug_tag"],
                field_name="evaluation.decision_kind_tagging.enable_python_derived_debug_tag",
            ),
        ),
        final_policy_set_selection=FinalPolicySetSelectionConfig(
            version=_require_text(selection["version"], field_name="evaluation.final_policy_set_selection.version"),
            include_random_legal_baseline_b0=_require_bool(
                selection["include_random_legal_baseline_b0"],
                field_name="evaluation.final_policy_set_selection.include_random_legal_baseline_b0",
            ),
            include_no_league_baseline_b1=_require_bool(
                selection["include_no_league_baseline_b1"],
                field_name="evaluation.final_policy_set_selection.include_no_league_baseline_b1",
            ),
            include_heuristic_public_b2_if_exists=_require_bool(
                selection["include_heuristic_public_b2_if_exists"],
                field_name="evaluation.final_policy_set_selection.include_heuristic_public_b2_if_exists",
            ),
            include_final_champion_snapshot=_require_bool(
                selection["include_final_champion_snapshot"],
                field_name="evaluation.final_policy_set_selection.include_final_champion_snapshot",
            ),
            include_spaced_snapshots_near_percent_updates=_require_int_list(
                selection["include_spaced_snapshots_near_percent_updates"],
                field_name="evaluation.final_policy_set_selection.include_spaced_snapshots_near_percent_updates",
            ),
            remaining_slots_strategy=_require_text(
                selection["remaining_slots_strategy"],
                field_name="evaluation.final_policy_set_selection.remaining_slots_strategy",
            ),
            fixed_anchor_set_v1=FixedAnchorSetConfig(
                required=_require_str_list(
                    fixed_anchor["required"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.required",
                ),
                optional_if_available=_require_str_list(
                    fixed_anchor["optional_if_available"],
                    field_name="evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available",
                ),
            ),
            seed_file=_require_text(
                selection["seed_file"], field_name="evaluation.final_policy_set_selection.seed_file"
            ),
            folding=_require_text(selection["folding"], field_name="evaluation.final_policy_set_selection.folding"),
            seat_swap=_require_bool(
                selection["seat_swap"], field_name="evaluation.final_policy_set_selection.seat_swap"
            ),
            tie_break=_require_text(
                selection["tie_break"], field_name="evaluation.final_policy_set_selection.tie_break"
            ),
        ),
    )


def _parse_reproducibility_config(body: dict[str, Any]) -> ReproducibilityConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "spec_bundle",
            "ids",
            "seed_derivation",
            "seed_files",
            "determinism_requirements",
            "legal_fingerprint",
        },
        context="reproducibility",
    )
    spec_bundle = _require_mapping(body["spec_bundle"], context="reproducibility.spec_bundle")
    ids = _require_mapping(body["ids"], context="reproducibility.ids")
    seed_derivation = _require_mapping(body["seed_derivation"], context="reproducibility.seed_derivation")
    seed_files = _require_mapping(body["seed_files"], context="reproducibility.seed_files")
    legal_fingerprint = _require_mapping(body["legal_fingerprint"], context="reproducibility.legal_fingerprint")
    _reject_unknown_keys(
        spec_bundle,
        allowed={"require_export_spec_bundle", "persist_in_manifest", "fail_on_spec_mismatch"},
        context="reproducibility.spec_bundle",
    )
    _reject_unknown_keys(
        ids,
        allowed={
            "run_id_hash",
            "config_hash",
            "spec_hash",
            "store_full_256_bit_ids",
            "store_short_64_bit_ids_for_filenames",
        },
        context="reproducibility.ids",
    )
    _reject_unknown_keys(
        seed_derivation,
        allowed={"base_seed64", "actor_seed_formula", "episode_seed_formula"},
        context="reproducibility.seed_derivation",
    )
    _reject_unknown_keys(
        legal_fingerprint,
        allowed={"version", "compute_in_rl_layer", "canonical_bytes", "replay_eval_mismatch_policy"},
        context="reproducibility.legal_fingerprint",
    )
    fail_on_spec_mismatch = _require_bool(
        spec_bundle["fail_on_spec_mismatch"],
        field_name="reproducibility.spec_bundle.fail_on_spec_mismatch",
    )
    require_fail_on_spec_mismatch(
        fail_on_spec_mismatch,
        source="reproducibility.spec_bundle.fail_on_spec_mismatch",
    )
    replay_eval_mismatch_policy = normalize_spec_mismatch_policy(
        legal_fingerprint["replay_eval_mismatch_policy"],
        source="reproducibility.legal_fingerprint.replay_eval_mismatch_policy",
    )
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
            fail_on_spec_mismatch=fail_on_spec_mismatch,
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
            version=_require_text(legal_fingerprint["version"], field_name="reproducibility.legal_fingerprint.version"),
            compute_in_rl_layer=_require_bool(
                legal_fingerprint["compute_in_rl_layer"],
                field_name="reproducibility.legal_fingerprint.compute_in_rl_layer",
            ),
            canonical_bytes=_require_str_list(
                legal_fingerprint["canonical_bytes"],
                field_name="reproducibility.legal_fingerprint.canonical_bytes",
            ),
            replay_eval_mismatch_policy=replay_eval_mismatch_policy,
        ),
    )


def _resolve_seed_sets(
    *,
    root: Path,
    league: LeagueConfig | None,
    evaluation: EvaluationConfig | None,
    reproducibility: ReproducibilityConfig | None,
) -> dict[str, Path]:
    seed_sets: dict[str, Path] = {}
    if evaluation is not None:
        for key, path in evaluation.seed_files.items():
            seed_sets[key] = _resolve_repo_path(root, path)
    if league is not None and league.promotion.seed_file.strip():
        seed_sets.setdefault("promotion_gate", _resolve_repo_path(root, league.promotion.seed_file))
    if reproducibility is not None:
        for key, path in reproducibility.seed_files.items():
            seed_sets.setdefault(key, _resolve_repo_path(root, path))
    return seed_sets


def _parse_seed_sets_override(*, root: Path, seed_sets_doc: Mapping[str, Any]) -> dict[str, Path]:
    return {
        _require_text(key, field_name="seed_sets.<key>"): _resolve_repo_path(
            root,
            _require_text(value, field_name=f"seed_sets.{key}"),
        )
        for key, value in seed_sets_doc.items()
    }


def _build_stack_config_from_component_doc(
    *,
    root: Path,
    component_root_doc: Mapping[str, Any],
    description: str,
    schema_version: int | None,
    seed_sets_override: Mapping[str, Any] | None = None,
    lock_intent: dict[str, Any] | None = None,
) -> StackConfig:
    _reject_unknown_keys(component_root_doc, allowed=_CONFIG_SECTION_KEYS, context="config")
    doc = dict(component_root_doc)

    experiment_doc = _require_mapping(doc["experiment"], context="experiment") if "experiment" in doc else None
    system_doc = _require_mapping(doc["system"], context="system") if "system" in doc else None
    model_doc = _require_mapping(doc["model"], context="model") if "model" in doc else None
    training_doc = _require_mapping(doc["training"], context="training") if "training" in doc else None
    environment_doc = _require_mapping(doc["environment"], context="environment") if "environment" in doc else None
    rewards_doc = _require_mapping(doc["rewards"], context="rewards") if "rewards" in doc else None
    league_doc = _require_mapping(doc["league"], context="league") if "league" in doc else None
    evaluation_doc = _require_mapping(doc["evaluation"], context="evaluation") if "evaluation" in doc else None
    reproducibility_doc = (
        _require_mapping(doc["reproducibility"], context="reproducibility") if "reproducibility" in doc else None
    )
    curriculum_doc = _require_mapping(doc["curriculum"], context="curriculum") if "curriculum" in doc else None

    experiment = _parse_experiment_config(experiment_doc) if experiment_doc is not None else None
    system = _parse_system_config(system_doc) if system_doc is not None else None
    model = _parse_model_config(model_doc) if model_doc is not None else None
    training = _parse_training_config(training_doc) if training_doc is not None else None
    environment = _parse_environment_config(environment_doc) if environment_doc is not None else None
    rewards = _parse_rewards_config(rewards_doc) if rewards_doc is not None else None
    curriculum = _parse_curriculum_config(curriculum_doc)
    league = _parse_league_config(league_doc) if league_doc is not None else None
    evaluation = _parse_evaluation_config(evaluation_doc) if evaluation_doc is not None else None
    reproducibility = _parse_reproducibility_config(reproducibility_doc) if reproducibility_doc is not None else None

    component_docs = {
        key: _require_mapping(value, context=key) for key, value in doc.items() if key in _CONFIG_SECTION_KEYS
    }
    seed_sets = (
        _parse_seed_sets_override(root=root, seed_sets_doc=seed_sets_override)
        if seed_sets_override is not None
        else _resolve_seed_sets(root=root, league=league, evaluation=evaluation, reproducibility=reproducibility)
    )

    return StackConfig(
        root=root,
        schema_version=schema_version,
        description=_require_text(description, field_name="description"),
        lock_intent={} if lock_intent is None else dict(lock_intent),
        components={},
        seed_sets=seed_sets,
        component_docs=component_docs,
        config=LockedConfig(
            experiment=experiment,
            system=system,
            model=model,
            training=training,
            environment=environment,
            rewards=rewards,
            curriculum=curriculum,
            league=league,
            evaluation=evaluation,
            reproducibility=reproducibility,
        ),
    )


def _load_canonical_stack_config(stack_file: Path) -> StackConfig:
    root = _resolve_repo_root(stack_file)
    payload = _load_json(stack_file)
    _reject_unknown_keys(payload, allowed=_CANONICAL_CONFIG_KEYS, context=str(stack_file))
    config_doc = _require_mapping(payload["config"], context="config")
    seed_sets_doc = _require_mapping(payload["seed_sets"], context="seed_sets") if "seed_sets" in payload else None
    schema_version = (
        _require_int(payload["schema_version"], field_name="schema_version", minimum=1)
        if "schema_version" in payload
        else None
    )
    description = _require_text(payload.get("description", stack_file.stem), field_name="description")
    return _build_stack_config_from_component_doc(
        root=root,
        component_root_doc=config_doc,
        description=description,
        schema_version=schema_version,
        seed_sets_override=seed_sets_doc,
        lock_intent={"canonical_config_payload": payload},
    )


def load_stack_config(stack_path: Path | str) -> StackConfig:
    stack_file = _resolve_legacy_config_path(Path(stack_path).resolve())
    if stack_file.suffix.lower() == ".json":
        return _load_canonical_stack_config(stack_file)

    root = _resolve_repo_root(stack_file)
    doc = _load_preset_document(stack_file)
    schema_version = (
        _require_int(doc["schema_version"], field_name="schema_version", minimum=1) if "schema_version" in doc else None
    )
    description = _require_text(doc.get("description", stack_file.stem), field_name="description")
    component_doc = {key: value for key, value in doc.items() if key in _CONFIG_SECTION_KEYS}
    return _build_stack_config_from_component_doc(
        root=root,
        component_root_doc=component_doc,
        description=description,
        schema_version=schema_version,
    )
