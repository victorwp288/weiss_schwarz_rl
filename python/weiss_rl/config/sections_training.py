"""Training stack config section parser."""

from __future__ import annotations

from typing import Any

from .models import (
    TrainingActionSurfaceConfig,
    TrainingCheckpointingConfig,
    TrainingConfig,
    TrainingExplorationConfig,
    TrainingOptimizerConfig,
    TrainingPpoConfig,
    TrainingPrecisionConfig,
    TrainingRolloutConfig,
    TrainingStructuredAuxConfig,
    TrainingStructuredMetricsConfig,
    TrainingStructuredWarmstartConfig,
    TrainingTeacherAuxConfig,
    TrainingVTraceConfig,
)
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_mapping,
    require_str_list,
    require_text,
)
from .sections_training_focus import (
    paired_swing_action_source,
    paired_swing_focus_fraction,
    paired_swing_focus_groups,
    paired_swing_focus_source_labels,
    trajectory_bc_focus_fraction,
    trajectory_bc_focus_groups,
    trajectory_bc_focus_source_labels,
    validate_paired_swing_focus_contract,
    validate_trajectory_bc_focus_contract,
)

TRAINING_ALGORITHMS = frozenset(
    {"impala_vtrace_gru", "impala_vtrace_ff", "ppo_lite_masked_v1", "structured_v2", "impala_vtrace_structured_v1"}
)
TRAINING_STRUCTURED_METRICS_MODES = frozenset({"off", "sampled", "full"})
TRAINING_TEACHER_AUX_MODES = frozenset({"off", "warmstart_only", "always"})
TRAINING_FIXED_OPPONENT_BACKENDS = frozenset({"python_scalar", "python_batched", "simulator_native"})
TRAINING_FIXED_MODEL_OPPONENT_ACTION_SELECTIONS = frozenset({"sample", "argmax"})
TRAINING_ACTOR_POLICY_BACKENDS = frozenset({"model", "heuristic_public"})
TRAINING_PUBLIC_HEURISTIC_PROFILES = frozenset({"base", "aggressive", "control"})
TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES = frozenset({"mixture", "cycle"})
TRAINING_ENTROPY_SCOPES = frozenset({"candidate", "family"})
TRAINING_PAIRED_SWING_CONFLICT_FILTERS = frozenset({"none", "current_state", "history"})
TRAINING_PAIRED_SWING_LOSS_SCOPES = frozenset({"row", "episode_mean", "label_mean"})
TRAINING_PAIRED_SWING_COMPARE_TO = frozenset({"negative", "top_other"})
TRAINING_PAIRED_OUTCOME_PREFERENCE_AGGREGATIONS = frozenset({"mean", "sum", "edge_mean"})
TRAINING_TRAJECTORY_RETENTION_SOURCES = frozenset(
    {"all_model", "champions", "hard_negatives", "recent", "warmup_snapshots"}
)


def parse_training_config(body: dict[str, Any]) -> TrainingConfig:
    reject_unknown_keys(
        body,
        allowed={
            "algorithm",
            "rollout",
            "optimizer",
            "exploration",
            "precision",
            "profile_timers",
            "torch_profiler",
            "checkpointing",
            "vtrace",
            "ppo",
            "structured_aux",
            "structured_warmstart",
            "structured_metrics",
            "teacher_aux",
            "action_surface",
            "fixed_opponent_backend",
            "fixed_model_opponent_action_selection",
            "actor_policy_backend",
            "actor_heuristic_fraction",
            "actor_heuristic_start_updates",
            "actor_heuristic_end_updates",
            "actor_heuristic_final_fraction",
            "train_on_heuristic_actor_rows",
            "diverse_opponent_actor_count",
            "diverse_model_actor_count",
            "diverse_opponent_batch_fraction",
            "diverse_opponent_batch_wait_ms",
            "heuristic_actor_hidden_state_tracking",
        },
        context="training",
    )
    rollout = require_mapping(body["rollout"], context="training.rollout")
    optimizer = require_mapping(body["optimizer"], context="training.optimizer")
    exploration = require_mapping(body["exploration"], context="training.exploration")
    precision = require_mapping(body["precision"], context="training.precision")
    checkpointing = require_mapping(body["checkpointing"], context="training.checkpointing")
    vtrace = require_mapping(body["vtrace"], context="training.vtrace")
    ppo = require_mapping(body.get("ppo", {}), context="training.ppo")
    structured_aux = require_mapping(body.get("structured_aux", {}), context="training.structured_aux")
    structured_warmstart = require_mapping(
        body.get("structured_warmstart", {}),
        context="training.structured_warmstart",
    )
    structured_metrics = require_mapping(
        body.get("structured_metrics", {}),
        context="training.structured_metrics",
    )
    teacher_aux = require_mapping(body.get("teacher_aux", {}), context="training.teacher_aux")
    action_surface = require_mapping(body.get("action_surface", {}), context="training.action_surface")
    actor_sampling_temperature = require_float(
        exploration.get("actor_sampling_temperature", 1.0),
        field_name="training.exploration.actor_sampling_temperature",
    )
    if actor_sampling_temperature <= 0.0:
        raise ValueError("training.exploration.actor_sampling_temperature must be > 0")

    reject_unknown_keys(rollout, allowed={"unroll_length", "batch_unrolls_per_update"}, context="training.rollout")
    reject_unknown_keys(
        optimizer,
        allowed={"name", "learning_rate", "grad_norm_clip", "value_loss_coef"},
        context="training.optimizer",
    )
    reject_unknown_keys(
        exploration,
        allowed={
            "entropy_coef",
            "entropy_anneal_to",
            "entropy_anneal_steps_updates",
            "entropy_scope",
            "actor_sampling_temperature",
        },
        context="training.exploration",
    )
    reject_unknown_keys(
        precision,
        allowed={"mixed_precision", "compile_learner", "compile_actor_inference", "masking_math_float32"},
        context="training.precision",
    )
    reject_unknown_keys(
        checkpointing,
        allowed={"checkpoint_interval_updates", "snapshot_interval_updates", "actor_reload_interval_updates"},
        context="training.checkpointing",
    )
    reject_unknown_keys(vtrace, allowed={"rho_bar", "c_bar"}, context="training.vtrace")
    reject_unknown_keys(
        ppo,
        allowed={"clip_epsilon", "value_clip_epsilon", "gae_lambda", "epochs", "target_kl", "normalize_advantages"},
        context="training.ppo",
    )
    reject_unknown_keys(
        structured_aux,
        allowed={
            "enabled",
            "teacher_family_coef",
            "teacher_slot_coef",
            "teacher_hand_coef",
            "teacher_move_source_coef",
            "teacher_attack_type_coef",
            "teacher_action_coef",
            "teacher_same_family_action_coef",
            "teacher_action_margin_coef",
            "teacher_action_margin",
            "teacher_same_family_action_margin_coef",
            "teacher_same_family_action_margin",
            "teacher_supervised_start_updates",
            "teacher_supervised_end_updates",
            "teacher_supervised_final_scale",
            "teacher_exact_action_families",
            "teacher_public_heuristic_coef",
            "teacher_public_heuristic_start_updates",
            "teacher_public_heuristic_end_updates",
            "teacher_public_heuristic_final_coef",
            "teacher_public_heuristic_temperature",
            "teacher_public_nonpass_over_pass_coef",
            "teacher_public_nonpass_over_pass_margin",
            "teacher_public_heuristic_families",
            "teacher_public_heuristic_profiles",
            "teacher_public_heuristic_profile_mode",
            "teacher_public_heuristic_profiles_end_updates",
            "policy_anchor_coef",
            "policy_anchor_top_action_coef",
            "policy_anchor_temperature",
            "trajectory_retention_coef",
            "trajectory_retention_policy_ids",
            "trajectory_retention_sources",
            "trajectory_bc_dataset_path",
            "trajectory_bc_every_updates",
            "trajectory_bc_aux_updates",
            "trajectory_bc_batch_episodes",
            "trajectory_bc_seed",
            "trajectory_bc_focus_source_labels",
            "trajectory_bc_focus_fraction",
            "trajectory_bc_focus_groups",
            "trajectory_bc_teacher_family_coef",
            "trajectory_bc_teacher_slot_coef",
            "trajectory_bc_teacher_move_source_coef",
            "trajectory_bc_teacher_attack_type_coef",
            "trajectory_bc_teacher_action_coef",
            "trajectory_bc_teacher_same_family_action_coef",
            "trajectory_bc_teacher_same_family_action_margin_coef",
            "trajectory_bc_teacher_same_family_action_margin",
            "paired_swing_dataset_path",
            "paired_swing_every_updates",
            "paired_swing_aux_updates",
            "paired_swing_batch_episodes",
            "paired_swing_seed",
            "paired_swing_focus_source_labels",
            "paired_swing_focus_fraction",
            "paired_swing_focus_groups",
            "paired_swing_margin",
            "paired_swing_coef",
            "paired_swing_positive_action_source",
            "paired_swing_negative_action_source",
            "paired_swing_conflict_filter",
            "paired_swing_loss_scope",
            "paired_swing_compare_to",
            "paired_outcome_preference_dataset_path",
            "paired_outcome_preference_every_updates",
            "paired_outcome_preference_aux_updates",
            "paired_outcome_preference_batch_episodes",
            "paired_outcome_preference_seed",
            "paired_outcome_preference_coef",
            "paired_outcome_preference_beta",
            "paired_outcome_preference_aggregation",
            "paired_outcome_preference_group_balance",
        },
        context="training.structured_aux",
    )
    reject_unknown_keys(
        structured_warmstart,
        allowed={
            "enabled",
            "updates",
            "teacher_family_coef",
            "teacher_slot_coef",
            "teacher_hand_coef",
            "teacher_move_source_coef",
            "teacher_attack_type_coef",
            "teacher_action_coef",
            "teacher_same_family_action_coef",
            "teacher_public_heuristic_coef",
            "teacher_public_heuristic_temperature",
            "teacher_public_heuristic_families",
            "teacher_public_heuristic_profiles",
            "teacher_public_heuristic_profile_mode",
            "teacher_public_heuristic_profiles_end_updates",
        },
        context="training.structured_warmstart",
    )
    reject_unknown_keys(structured_metrics, allowed={"mode"}, context="training.structured_metrics")
    reject_unknown_keys(teacher_aux, allowed={"mode"}, context="training.teacher_aux")
    reject_unknown_keys(
        action_surface,
        allowed={
            "mulligan_force_confirm_after_select",
            "force_pass_over_main_move_only",
            "main_move_only_max_consecutive",
            "force_attack_over_pass_when_attack_legal",
        },
        context="training.action_surface",
    )

    profile_timers = require_bool(body.get("profile_timers", False), field_name="training.profile_timers")
    torch_profiler = require_bool(body.get("torch_profiler", False), field_name="training.torch_profiler")
    fixed_opponent_backend = require_choice(
        body.get("fixed_opponent_backend", "python_scalar"),
        field_name="training.fixed_opponent_backend",
        allowed=TRAINING_FIXED_OPPONENT_BACKENDS,
    )
    fixed_model_opponent_action_selection = require_choice(
        body.get("fixed_model_opponent_action_selection", "sample"),
        field_name="training.fixed_model_opponent_action_selection",
        allowed=TRAINING_FIXED_MODEL_OPPONENT_ACTION_SELECTIONS,
    )
    actor_policy_backend = require_choice(
        body.get("actor_policy_backend", "model"),
        field_name="training.actor_policy_backend",
        allowed=TRAINING_ACTOR_POLICY_BACKENDS,
    )
    actor_heuristic_fraction = require_float(
        body.get("actor_heuristic_fraction", 1.0),
        field_name="training.actor_heuristic_fraction",
    )
    if actor_heuristic_fraction < 0.0 or actor_heuristic_fraction > 1.0:
        raise ValueError(
            f"training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive, got {actor_heuristic_fraction}"
        )
    actor_heuristic_start_updates = require_int(
        body.get("actor_heuristic_start_updates", 0),
        field_name="training.actor_heuristic_start_updates",
        minimum=0,
    )
    actor_heuristic_end_updates = require_int(
        body.get("actor_heuristic_end_updates", -1),
        field_name="training.actor_heuristic_end_updates",
        minimum=-1,
    )
    if actor_heuristic_end_updates >= 0 and actor_heuristic_end_updates < actor_heuristic_start_updates:
        raise ValueError("training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates")
    actor_heuristic_final_fraction = require_float(
        body.get("actor_heuristic_final_fraction", actor_heuristic_fraction),
        field_name="training.actor_heuristic_final_fraction",
    )
    if actor_heuristic_final_fraction < 0.0 or actor_heuristic_final_fraction > 1.0:
        raise ValueError(
            "training.actor_heuristic_final_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {actor_heuristic_final_fraction}"
        )
    heuristic_actor_hidden_state_tracking = require_bool(
        body.get("heuristic_actor_hidden_state_tracking", True),
        field_name="training.heuristic_actor_hidden_state_tracking",
    )
    train_on_heuristic_actor_rows = require_bool(
        body.get("train_on_heuristic_actor_rows", True),
        field_name="training.train_on_heuristic_actor_rows",
    )
    diverse_opponent_actor_count = require_int(
        body.get("diverse_opponent_actor_count", 0),
        field_name="training.diverse_opponent_actor_count",
        minimum=0,
    )
    diverse_model_actor_count = require_int(
        body.get("diverse_model_actor_count", 0),
        field_name="training.diverse_model_actor_count",
        minimum=0,
    )
    diverse_opponent_batch_fraction = require_float(
        body.get("diverse_opponent_batch_fraction", 0.0),
        field_name="training.diverse_opponent_batch_fraction",
    )
    if diverse_opponent_batch_fraction < 0.0 or diverse_opponent_batch_fraction > 1.0:
        raise ValueError(
            "training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {diverse_opponent_batch_fraction}"
        )
    diverse_opponent_batch_wait_ms = require_int(
        body.get("diverse_opponent_batch_wait_ms", 0),
        field_name="training.diverse_opponent_batch_wait_ms",
        minimum=0,
    )
    structured_aux_public_temperature = require_float(
        structured_aux.get("teacher_public_heuristic_temperature", 32.0),
        field_name="training.structured_aux.teacher_public_heuristic_temperature",
    )
    if structured_aux_public_temperature <= 0.0:
        raise ValueError("training.structured_aux.teacher_public_heuristic_temperature must be > 0")
    structured_warmstart_public_temperature = require_float(
        structured_warmstart.get("teacher_public_heuristic_temperature", 32.0),
        field_name="training.structured_warmstart.teacher_public_heuristic_temperature",
    )
    if structured_warmstart_public_temperature <= 0.0:
        raise ValueError("training.structured_warmstart.teacher_public_heuristic_temperature must be > 0")
    structured_aux_action_margin_coef = require_float(
        structured_aux.get("teacher_action_margin_coef", 0.0),
        field_name="training.structured_aux.teacher_action_margin_coef",
    )
    if structured_aux_action_margin_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_action_margin_coef must be >= 0.0")
    structured_aux_action_margin = require_float(
        structured_aux.get("teacher_action_margin", 0.5),
        field_name="training.structured_aux.teacher_action_margin",
    )
    if structured_aux_action_margin < 0.0:
        raise ValueError("training.structured_aux.teacher_action_margin must be >= 0.0")
    structured_aux_same_family_action_margin_coef = require_float(
        structured_aux.get("teacher_same_family_action_margin_coef", 0.0),
        field_name="training.structured_aux.teacher_same_family_action_margin_coef",
    )
    if structured_aux_same_family_action_margin_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_same_family_action_margin_coef must be >= 0.0")
    structured_aux_same_family_action_margin = require_float(
        structured_aux.get("teacher_same_family_action_margin", 0.5),
        field_name="training.structured_aux.teacher_same_family_action_margin",
    )
    if structured_aux_same_family_action_margin < 0.0:
        raise ValueError("training.structured_aux.teacher_same_family_action_margin must be >= 0.0")
    structured_aux_supervised_start_updates = require_int(
        structured_aux.get("teacher_supervised_start_updates", 0),
        field_name="training.structured_aux.teacher_supervised_start_updates",
        minimum=0,
    )
    structured_aux_supervised_end_updates = require_int(
        structured_aux.get("teacher_supervised_end_updates", -1),
        field_name="training.structured_aux.teacher_supervised_end_updates",
        minimum=-1,
    )
    if (
        structured_aux_supervised_end_updates >= 0
        and structured_aux_supervised_end_updates < structured_aux_supervised_start_updates
    ):
        raise ValueError(
            "training.structured_aux.teacher_supervised_end_updates must be >= "
            "training.structured_aux.teacher_supervised_start_updates"
        )
    structured_aux_supervised_final_scale = require_float(
        structured_aux.get("teacher_supervised_final_scale", 1.0),
        field_name="training.structured_aux.teacher_supervised_final_scale",
    )
    if structured_aux_supervised_final_scale < 0.0:
        raise ValueError("training.structured_aux.teacher_supervised_final_scale must be >= 0.0")
    structured_aux_public_nonpass_coef = require_float(
        structured_aux.get("teacher_public_nonpass_over_pass_coef", 0.0),
        field_name="training.structured_aux.teacher_public_nonpass_over_pass_coef",
    )
    if structured_aux_public_nonpass_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_public_nonpass_over_pass_coef must be >= 0.0")
    structured_aux_public_nonpass_margin = require_float(
        structured_aux.get("teacher_public_nonpass_over_pass_margin", 0.5),
        field_name="training.structured_aux.teacher_public_nonpass_over_pass_margin",
    )
    if structured_aux_public_nonpass_margin < 0.0:
        raise ValueError("training.structured_aux.teacher_public_nonpass_over_pass_margin must be >= 0.0")
    structured_aux_public_profiles = tuple(
        name.strip().lower()
        for name in require_str_list(
            structured_aux.get("teacher_public_heuristic_profiles", []),
            field_name="training.structured_aux.teacher_public_heuristic_profiles",
        )
        if name.strip()
    )
    invalid_aux_public_profiles = sorted(set(structured_aux_public_profiles) - TRAINING_PUBLIC_HEURISTIC_PROFILES)
    if invalid_aux_public_profiles:
        raise ValueError(
            "training.structured_aux.teacher_public_heuristic_profiles contains unsupported profiles: "
            + ", ".join(invalid_aux_public_profiles)
        )
    structured_warmstart_public_profiles = tuple(
        name.strip().lower()
        for name in require_str_list(
            structured_warmstart.get("teacher_public_heuristic_profiles", []),
            field_name="training.structured_warmstart.teacher_public_heuristic_profiles",
        )
        if name.strip()
    )
    invalid_warmstart_public_profiles = sorted(
        set(structured_warmstart_public_profiles) - TRAINING_PUBLIC_HEURISTIC_PROFILES
    )
    if invalid_warmstart_public_profiles:
        raise ValueError(
            "training.structured_warmstart.teacher_public_heuristic_profiles contains unsupported profiles: "
            + ", ".join(invalid_warmstart_public_profiles)
        )
    structured_aux_public_profile_mode = require_choice(
        structured_aux.get("teacher_public_heuristic_profile_mode", "mixture"),
        field_name="training.structured_aux.teacher_public_heuristic_profile_mode",
        allowed=TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
    )
    structured_aux_public_start_updates = require_int(
        structured_aux.get("teacher_public_heuristic_start_updates", 0),
        field_name="training.structured_aux.teacher_public_heuristic_start_updates",
        minimum=0,
    )
    structured_aux_public_end_updates = require_int(
        structured_aux.get("teacher_public_heuristic_end_updates", -1),
        field_name="training.structured_aux.teacher_public_heuristic_end_updates",
        minimum=-1,
    )
    if (
        structured_aux_public_end_updates >= 0
        and structured_aux_public_end_updates < structured_aux_public_start_updates
    ):
        raise ValueError(
            "training.structured_aux.teacher_public_heuristic_end_updates must be >= "
            "training.structured_aux.teacher_public_heuristic_start_updates"
        )
    structured_aux_public_final_coef = require_float(
        structured_aux.get(
            "teacher_public_heuristic_final_coef",
            structured_aux.get("teacher_public_heuristic_coef", 0.0),
        ),
        field_name="training.structured_aux.teacher_public_heuristic_final_coef",
    )
    if structured_aux_public_final_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_public_heuristic_final_coef must be >= 0.0")
    structured_aux_public_profiles_end_updates = require_int(
        structured_aux.get("teacher_public_heuristic_profiles_end_updates", -1),
        field_name="training.structured_aux.teacher_public_heuristic_profiles_end_updates",
        minimum=-1,
    )
    structured_aux_policy_anchor_coef = require_float(
        structured_aux.get("policy_anchor_coef", 0.0),
        field_name="training.structured_aux.policy_anchor_coef",
    )
    if structured_aux_policy_anchor_coef < 0.0:
        raise ValueError("training.structured_aux.policy_anchor_coef must be >= 0.0")
    structured_aux_policy_anchor_top_action_coef = require_float(
        structured_aux.get("policy_anchor_top_action_coef", 0.0),
        field_name="training.structured_aux.policy_anchor_top_action_coef",
    )
    if structured_aux_policy_anchor_top_action_coef < 0.0:
        raise ValueError("training.structured_aux.policy_anchor_top_action_coef must be >= 0.0")
    structured_aux_policy_anchor_temperature = require_float(
        structured_aux.get("policy_anchor_temperature", 1.0),
        field_name="training.structured_aux.policy_anchor_temperature",
    )
    if structured_aux_policy_anchor_temperature <= 0.0:
        raise ValueError("training.structured_aux.policy_anchor_temperature must be > 0")
    structured_aux_trajectory_retention_coef = require_float(
        structured_aux.get("trajectory_retention_coef", 0.0),
        field_name="training.structured_aux.trajectory_retention_coef",
    )
    if structured_aux_trajectory_retention_coef < 0.0:
        raise ValueError("training.structured_aux.trajectory_retention_coef must be >= 0.0")
    structured_aux_trajectory_retention_policy_ids = tuple(
        str(policy_id).strip()
        for policy_id in require_str_list(
            structured_aux.get("trajectory_retention_policy_ids", []),
            field_name="training.structured_aux.trajectory_retention_policy_ids",
        )
        if str(policy_id).strip()
    )
    structured_aux_trajectory_retention_sources = tuple(
        source.strip().lower()
        for source in require_str_list(
            structured_aux.get("trajectory_retention_sources", ["champions"]),
            field_name="training.structured_aux.trajectory_retention_sources",
        )
        if source.strip()
    )
    invalid_retention_sources = sorted(
        set(structured_aux_trajectory_retention_sources) - TRAINING_TRAJECTORY_RETENTION_SOURCES
    )
    if invalid_retention_sources:
        raise ValueError(
            "training.structured_aux.trajectory_retention_sources contains unsupported sources: "
            + ", ".join(invalid_retention_sources)
        )
    structured_aux_trajectory_bc_focus_source_labels = trajectory_bc_focus_source_labels(structured_aux)
    structured_aux_trajectory_bc_focus_fraction = trajectory_bc_focus_fraction(structured_aux)
    structured_aux_trajectory_bc_focus_groups = trajectory_bc_focus_groups(structured_aux)
    validate_trajectory_bc_focus_contract(
        source_labels=structured_aux_trajectory_bc_focus_source_labels,
        fraction=structured_aux_trajectory_bc_focus_fraction,
        groups=structured_aux_trajectory_bc_focus_groups,
    )
    structured_aux_paired_swing_focus_source_labels = paired_swing_focus_source_labels(structured_aux)
    structured_aux_paired_swing_focus_fraction = paired_swing_focus_fraction(structured_aux)
    structured_aux_paired_swing_focus_groups = paired_swing_focus_groups(structured_aux)
    validate_paired_swing_focus_contract(
        source_labels=structured_aux_paired_swing_focus_source_labels,
        fraction=structured_aux_paired_swing_focus_fraction,
        groups=structured_aux_paired_swing_focus_groups,
    )
    structured_aux_paired_swing_margin = require_float(
        structured_aux.get("paired_swing_margin", 0.35),
        field_name="training.structured_aux.paired_swing_margin",
    )
    if structured_aux_paired_swing_margin < 0.0:
        raise ValueError("training.structured_aux.paired_swing_margin must be >= 0.0")
    structured_aux_paired_swing_coef = require_float(
        structured_aux.get("paired_swing_coef", 0.08),
        field_name="training.structured_aux.paired_swing_coef",
    )
    if structured_aux_paired_swing_coef < 0.0:
        raise ValueError("training.structured_aux.paired_swing_coef must be >= 0.0")
    structured_aux_paired_swing_positive_source = paired_swing_action_source(
        structured_aux,
        key="paired_swing_positive_action_source",
        default="teacher_action",
    )
    structured_aux_paired_swing_negative_source = paired_swing_action_source(
        structured_aux,
        key="paired_swing_negative_action_source",
        default="actions",
    )
    if structured_aux_paired_swing_positive_source == structured_aux_paired_swing_negative_source:
        raise ValueError(
            "training.structured_aux.paired_swing_positive_action_source and "
            "paired_swing_negative_action_source must differ"
        )
    structured_aux_paired_swing_conflict_filter = require_choice(
        structured_aux.get("paired_swing_conflict_filter", "none"),
        field_name="training.structured_aux.paired_swing_conflict_filter",
        allowed=TRAINING_PAIRED_SWING_CONFLICT_FILTERS,
    )
    structured_aux_paired_swing_loss_scope = require_choice(
        structured_aux.get("paired_swing_loss_scope", "row"),
        field_name="training.structured_aux.paired_swing_loss_scope",
        allowed=TRAINING_PAIRED_SWING_LOSS_SCOPES,
    )
    structured_aux_paired_swing_compare_to = require_choice(
        structured_aux.get("paired_swing_compare_to", "negative"),
        field_name="training.structured_aux.paired_swing_compare_to",
        allowed=TRAINING_PAIRED_SWING_COMPARE_TO,
    )
    structured_aux_paired_outcome_preference_coef = require_float(
        structured_aux.get("paired_outcome_preference_coef", 0.05),
        field_name="training.structured_aux.paired_outcome_preference_coef",
    )
    if structured_aux_paired_outcome_preference_coef < 0.0:
        raise ValueError("training.structured_aux.paired_outcome_preference_coef must be >= 0.0")
    structured_aux_paired_outcome_preference_beta = require_float(
        structured_aux.get("paired_outcome_preference_beta", 0.1),
        field_name="training.structured_aux.paired_outcome_preference_beta",
    )
    if structured_aux_paired_outcome_preference_beta <= 0.0:
        raise ValueError("training.structured_aux.paired_outcome_preference_beta must be > 0.0")
    structured_aux_paired_outcome_preference_aggregation = require_choice(
        structured_aux.get("paired_outcome_preference_aggregation", "mean"),
        field_name="training.structured_aux.paired_outcome_preference_aggregation",
        allowed=TRAINING_PAIRED_OUTCOME_PREFERENCE_AGGREGATIONS,
    )
    structured_warmstart_public_profile_mode = require_choice(
        structured_warmstart.get("teacher_public_heuristic_profile_mode", "mixture"),
        field_name="training.structured_warmstart.teacher_public_heuristic_profile_mode",
        allowed=TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
    )
    structured_warmstart_public_profiles_end_updates = require_int(
        structured_warmstart.get("teacher_public_heuristic_profiles_end_updates", -1),
        field_name="training.structured_warmstart.teacher_public_heuristic_profiles_end_updates",
        minimum=-1,
    )

    return TrainingConfig(
        algorithm=require_choice(body["algorithm"], field_name="training.algorithm", allowed=TRAINING_ALGORITHMS),
        rollout=TrainingRolloutConfig(
            unroll_length=require_int(rollout["unroll_length"], field_name="training.rollout.unroll_length", minimum=1),
            batch_unrolls_per_update=require_int(
                rollout["batch_unrolls_per_update"],
                field_name="training.rollout.batch_unrolls_per_update",
                minimum=1,
            ),
        ),
        optimizer=TrainingOptimizerConfig(
            name=require_text(optimizer["name"], field_name="training.optimizer.name"),
            learning_rate=require_float(optimizer["learning_rate"], field_name="training.optimizer.learning_rate"),
            grad_norm_clip=require_float(optimizer["grad_norm_clip"], field_name="training.optimizer.grad_norm_clip"),
            value_loss_coef=require_float(
                optimizer["value_loss_coef"], field_name="training.optimizer.value_loss_coef"
            ),
        ),
        exploration=TrainingExplorationConfig(
            entropy_coef=require_float(exploration["entropy_coef"], field_name="training.exploration.entropy_coef"),
            entropy_anneal_to=require_float(
                exploration["entropy_anneal_to"], field_name="training.exploration.entropy_anneal_to"
            ),
            entropy_anneal_steps_updates=require_int(
                exploration["entropy_anneal_steps_updates"],
                field_name="training.exploration.entropy_anneal_steps_updates",
                minimum=1,
            ),
            entropy_scope=require_choice(
                exploration.get("entropy_scope", "candidate"),
                field_name="training.exploration.entropy_scope",
                allowed=TRAINING_ENTROPY_SCOPES,
            ),
            actor_sampling_temperature=actor_sampling_temperature,
        ),
        precision=TrainingPrecisionConfig(
            mixed_precision=require_bool(precision["mixed_precision"], field_name="training.precision.mixed_precision"),
            compile_learner=require_bool(precision["compile_learner"], field_name="training.precision.compile_learner"),
            compile_actor_inference=require_bool(
                precision.get("compile_actor_inference", False),
                field_name="training.precision.compile_actor_inference",
            ),
            masking_math_float32=require_bool(
                precision["masking_math_float32"],
                field_name="training.precision.masking_math_float32",
            ),
        ),
        profile_timers=profile_timers,
        torch_profiler=torch_profiler,
        checkpointing=TrainingCheckpointingConfig(
            checkpoint_interval_updates=require_int(
                checkpointing["checkpoint_interval_updates"],
                field_name="training.checkpointing.checkpoint_interval_updates",
                minimum=1,
            ),
            snapshot_interval_updates=require_int(
                checkpointing["snapshot_interval_updates"],
                field_name="training.checkpointing.snapshot_interval_updates",
                minimum=1,
            ),
            actor_reload_interval_updates=require_int(
                checkpointing["actor_reload_interval_updates"],
                field_name="training.checkpointing.actor_reload_interval_updates",
                minimum=1,
            ),
        ),
        vtrace=TrainingVTraceConfig(
            rho_bar=require_float(vtrace["rho_bar"], field_name="training.vtrace.rho_bar"),
            c_bar=require_float(vtrace["c_bar"], field_name="training.vtrace.c_bar"),
        ),
        ppo=TrainingPpoConfig(
            clip_epsilon=require_float(ppo.get("clip_epsilon", 0.2), field_name="training.ppo.clip_epsilon"),
            value_clip_epsilon=require_float(
                ppo.get("value_clip_epsilon", 0.2),
                field_name="training.ppo.value_clip_epsilon",
            ),
            gae_lambda=require_float(ppo.get("gae_lambda", 0.95), field_name="training.ppo.gae_lambda"),
            epochs=require_int(ppo.get("epochs", 4), field_name="training.ppo.epochs", minimum=1),
            target_kl=require_float(ppo.get("target_kl", 0.0), field_name="training.ppo.target_kl"),
            normalize_advantages=require_bool(
                ppo.get("normalize_advantages", True),
                field_name="training.ppo.normalize_advantages",
            ),
        ),
        structured_aux=TrainingStructuredAuxConfig(
            enabled=require_bool(
                structured_aux.get("enabled", False),
                field_name="training.structured_aux.enabled",
            ),
            teacher_family_coef=require_float(
                structured_aux.get("teacher_family_coef", 0.0),
                field_name="training.structured_aux.teacher_family_coef",
            ),
            teacher_slot_coef=require_float(
                structured_aux.get("teacher_slot_coef", 0.0),
                field_name="training.structured_aux.teacher_slot_coef",
            ),
            teacher_hand_coef=require_float(
                structured_aux.get("teacher_hand_coef", 0.0),
                field_name="training.structured_aux.teacher_hand_coef",
            ),
            teacher_move_source_coef=require_float(
                structured_aux.get("teacher_move_source_coef", 0.0),
                field_name="training.structured_aux.teacher_move_source_coef",
            ),
            teacher_attack_type_coef=require_float(
                structured_aux.get("teacher_attack_type_coef", 0.0),
                field_name="training.structured_aux.teacher_attack_type_coef",
            ),
            teacher_action_coef=require_float(
                structured_aux.get("teacher_action_coef", 0.0),
                field_name="training.structured_aux.teacher_action_coef",
            ),
            teacher_same_family_action_coef=require_float(
                structured_aux.get("teacher_same_family_action_coef", 0.0),
                field_name="training.structured_aux.teacher_same_family_action_coef",
            ),
            teacher_action_margin_coef=structured_aux_action_margin_coef,
            teacher_action_margin=structured_aux_action_margin,
            teacher_same_family_action_margin_coef=structured_aux_same_family_action_margin_coef,
            teacher_same_family_action_margin=structured_aux_same_family_action_margin,
            teacher_supervised_start_updates=structured_aux_supervised_start_updates,
            teacher_supervised_end_updates=structured_aux_supervised_end_updates,
            teacher_supervised_final_scale=structured_aux_supervised_final_scale,
            teacher_exact_action_families=require_str_list(
                structured_aux.get("teacher_exact_action_families", []),
                field_name="training.structured_aux.teacher_exact_action_families",
            ),
            teacher_public_heuristic_coef=require_float(
                structured_aux.get("teacher_public_heuristic_coef", 0.0),
                field_name="training.structured_aux.teacher_public_heuristic_coef",
            ),
            teacher_public_heuristic_start_updates=structured_aux_public_start_updates,
            teacher_public_heuristic_end_updates=structured_aux_public_end_updates,
            teacher_public_heuristic_final_coef=require_float(
                structured_aux_public_final_coef,
                field_name="training.structured_aux.teacher_public_heuristic_final_coef",
            ),
            teacher_public_heuristic_temperature=structured_aux_public_temperature,
            teacher_public_nonpass_over_pass_coef=structured_aux_public_nonpass_coef,
            teacher_public_nonpass_over_pass_margin=structured_aux_public_nonpass_margin,
            teacher_public_heuristic_families=require_str_list(
                structured_aux.get("teacher_public_heuristic_families", []),
                field_name="training.structured_aux.teacher_public_heuristic_families",
            ),
            teacher_public_heuristic_profiles=structured_aux_public_profiles,
            teacher_public_heuristic_profile_mode=structured_aux_public_profile_mode,
            teacher_public_heuristic_profiles_end_updates=structured_aux_public_profiles_end_updates,
            policy_anchor_coef=structured_aux_policy_anchor_coef,
            policy_anchor_top_action_coef=structured_aux_policy_anchor_top_action_coef,
            policy_anchor_temperature=structured_aux_policy_anchor_temperature,
            trajectory_retention_coef=structured_aux_trajectory_retention_coef,
            trajectory_retention_policy_ids=structured_aux_trajectory_retention_policy_ids,
            trajectory_retention_sources=structured_aux_trajectory_retention_sources,
            trajectory_bc_dataset_path=str(structured_aux.get("trajectory_bc_dataset_path", "")).strip(),
            trajectory_bc_every_updates=require_int(
                structured_aux.get("trajectory_bc_every_updates", 0),
                field_name="training.structured_aux.trajectory_bc_every_updates",
                minimum=0,
            ),
            trajectory_bc_aux_updates=require_int(
                structured_aux.get("trajectory_bc_aux_updates", 1),
                field_name="training.structured_aux.trajectory_bc_aux_updates",
                minimum=1,
            ),
            trajectory_bc_batch_episodes=require_int(
                structured_aux.get("trajectory_bc_batch_episodes", 8),
                field_name="training.structured_aux.trajectory_bc_batch_episodes",
                minimum=1,
            ),
            trajectory_bc_seed=require_int(
                structured_aux.get("trajectory_bc_seed", 20260516),
                field_name="training.structured_aux.trajectory_bc_seed",
                minimum=0,
            ),
            trajectory_bc_focus_source_labels=structured_aux_trajectory_bc_focus_source_labels,
            trajectory_bc_focus_fraction=structured_aux_trajectory_bc_focus_fraction,
            trajectory_bc_focus_groups=structured_aux_trajectory_bc_focus_groups,
            trajectory_bc_teacher_family_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_family_coef", 0.05),
                field_name="training.structured_aux.trajectory_bc_teacher_family_coef",
            ),
            trajectory_bc_teacher_slot_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_slot_coef", 0.05),
                field_name="training.structured_aux.trajectory_bc_teacher_slot_coef",
            ),
            trajectory_bc_teacher_move_source_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_move_source_coef", 0.02),
                field_name="training.structured_aux.trajectory_bc_teacher_move_source_coef",
            ),
            trajectory_bc_teacher_attack_type_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_attack_type_coef", 0.02),
                field_name="training.structured_aux.trajectory_bc_teacher_attack_type_coef",
            ),
            trajectory_bc_teacher_action_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_action_coef", 0.20),
                field_name="training.structured_aux.trajectory_bc_teacher_action_coef",
            ),
            trajectory_bc_teacher_same_family_action_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_same_family_action_coef", 0.60),
                field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_coef",
            ),
            trajectory_bc_teacher_same_family_action_margin_coef=require_float(
                structured_aux.get("trajectory_bc_teacher_same_family_action_margin_coef", 0.10),
                field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_margin_coef",
            ),
            trajectory_bc_teacher_same_family_action_margin=require_float(
                structured_aux.get("trajectory_bc_teacher_same_family_action_margin", 0.5),
                field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_margin",
            ),
            paired_swing_dataset_path=str(structured_aux.get("paired_swing_dataset_path", "")).strip(),
            paired_swing_every_updates=require_int(
                structured_aux.get("paired_swing_every_updates", 0),
                field_name="training.structured_aux.paired_swing_every_updates",
                minimum=0,
            ),
            paired_swing_aux_updates=require_int(
                structured_aux.get("paired_swing_aux_updates", 1),
                field_name="training.structured_aux.paired_swing_aux_updates",
                minimum=1,
            ),
            paired_swing_batch_episodes=require_int(
                structured_aux.get("paired_swing_batch_episodes", 8),
                field_name="training.structured_aux.paired_swing_batch_episodes",
                minimum=1,
            ),
            paired_swing_seed=require_int(
                structured_aux.get("paired_swing_seed", 20260519),
                field_name="training.structured_aux.paired_swing_seed",
                minimum=0,
            ),
            paired_swing_focus_source_labels=structured_aux_paired_swing_focus_source_labels,
            paired_swing_focus_fraction=structured_aux_paired_swing_focus_fraction,
            paired_swing_focus_groups=structured_aux_paired_swing_focus_groups,
            paired_swing_margin=structured_aux_paired_swing_margin,
            paired_swing_coef=structured_aux_paired_swing_coef,
            paired_swing_positive_action_source=structured_aux_paired_swing_positive_source,
            paired_swing_negative_action_source=structured_aux_paired_swing_negative_source,
            paired_swing_conflict_filter=structured_aux_paired_swing_conflict_filter,
            paired_swing_loss_scope=structured_aux_paired_swing_loss_scope,
            paired_swing_compare_to=structured_aux_paired_swing_compare_to,
            paired_outcome_preference_dataset_path=str(
                structured_aux.get("paired_outcome_preference_dataset_path", "")
            ).strip(),
            paired_outcome_preference_every_updates=require_int(
                structured_aux.get("paired_outcome_preference_every_updates", 0),
                field_name="training.structured_aux.paired_outcome_preference_every_updates",
                minimum=0,
            ),
            paired_outcome_preference_aux_updates=require_int(
                structured_aux.get("paired_outcome_preference_aux_updates", 1),
                field_name="training.structured_aux.paired_outcome_preference_aux_updates",
                minimum=1,
            ),
            paired_outcome_preference_batch_episodes=require_int(
                structured_aux.get("paired_outcome_preference_batch_episodes", 8),
                field_name="training.structured_aux.paired_outcome_preference_batch_episodes",
                minimum=1,
            ),
            paired_outcome_preference_seed=require_int(
                structured_aux.get("paired_outcome_preference_seed", 20260520),
                field_name="training.structured_aux.paired_outcome_preference_seed",
                minimum=0,
            ),
            paired_outcome_preference_coef=structured_aux_paired_outcome_preference_coef,
            paired_outcome_preference_beta=structured_aux_paired_outcome_preference_beta,
            paired_outcome_preference_aggregation=structured_aux_paired_outcome_preference_aggregation,
            paired_outcome_preference_group_balance=require_bool(
                structured_aux.get("paired_outcome_preference_group_balance", False),
                field_name="training.structured_aux.paired_outcome_preference_group_balance",
            ),
        ),
        structured_warmstart=TrainingStructuredWarmstartConfig(
            enabled=require_bool(
                structured_warmstart.get("enabled", False),
                field_name="training.structured_warmstart.enabled",
            ),
            updates=require_int(
                structured_warmstart.get("updates", 0),
                field_name="training.structured_warmstart.updates",
                minimum=0,
            ),
            teacher_family_coef=require_float(
                structured_warmstart.get("teacher_family_coef", 0.0),
                field_name="training.structured_warmstart.teacher_family_coef",
            ),
            teacher_slot_coef=require_float(
                structured_warmstart.get("teacher_slot_coef", 0.0),
                field_name="training.structured_warmstart.teacher_slot_coef",
            ),
            teacher_hand_coef=require_float(
                structured_warmstart.get("teacher_hand_coef", 0.0),
                field_name="training.structured_warmstart.teacher_hand_coef",
            ),
            teacher_move_source_coef=require_float(
                structured_warmstart.get("teacher_move_source_coef", 0.0),
                field_name="training.structured_warmstart.teacher_move_source_coef",
            ),
            teacher_attack_type_coef=require_float(
                structured_warmstart.get("teacher_attack_type_coef", 0.0),
                field_name="training.structured_warmstart.teacher_attack_type_coef",
            ),
            teacher_action_coef=require_float(
                structured_warmstart.get("teacher_action_coef", 0.0),
                field_name="training.structured_warmstart.teacher_action_coef",
            ),
            teacher_same_family_action_coef=require_float(
                structured_warmstart.get("teacher_same_family_action_coef", 0.0),
                field_name="training.structured_warmstart.teacher_same_family_action_coef",
            ),
            teacher_public_heuristic_coef=require_float(
                structured_warmstart.get("teacher_public_heuristic_coef", 0.0),
                field_name="training.structured_warmstart.teacher_public_heuristic_coef",
            ),
            teacher_public_heuristic_temperature=structured_warmstart_public_temperature,
            teacher_public_heuristic_families=require_str_list(
                structured_warmstart.get("teacher_public_heuristic_families", []),
                field_name="training.structured_warmstart.teacher_public_heuristic_families",
            ),
            teacher_public_heuristic_profiles=structured_warmstart_public_profiles,
            teacher_public_heuristic_profile_mode=structured_warmstart_public_profile_mode,
            teacher_public_heuristic_profiles_end_updates=structured_warmstart_public_profiles_end_updates,
        ),
        structured_metrics=TrainingStructuredMetricsConfig(
            mode=require_choice(
                structured_metrics.get("mode", "off"),
                field_name="training.structured_metrics.mode",
                allowed=TRAINING_STRUCTURED_METRICS_MODES,
            ),
        ),
        teacher_aux=TrainingTeacherAuxConfig(
            mode=require_choice(
                teacher_aux.get("mode", "always"),
                field_name="training.teacher_aux.mode",
                allowed=TRAINING_TEACHER_AUX_MODES,
            ),
        ),
        action_surface=TrainingActionSurfaceConfig(
            mulligan_force_confirm_after_select=require_bool(
                action_surface.get("mulligan_force_confirm_after_select", False),
                field_name="training.action_surface.mulligan_force_confirm_after_select",
            ),
            force_pass_over_main_move_only=require_bool(
                action_surface.get("force_pass_over_main_move_only", False),
                field_name="training.action_surface.force_pass_over_main_move_only",
            ),
            main_move_only_max_consecutive=require_int(
                action_surface.get("main_move_only_max_consecutive", 0),
                field_name="training.action_surface.main_move_only_max_consecutive",
                minimum=0,
            ),
            force_attack_over_pass_when_attack_legal=require_bool(
                action_surface.get("force_attack_over_pass_when_attack_legal", False),
                field_name="training.action_surface.force_attack_over_pass_when_attack_legal",
            ),
        ),
        fixed_opponent_backend=fixed_opponent_backend,
        fixed_model_opponent_action_selection=fixed_model_opponent_action_selection,
        actor_policy_backend=actor_policy_backend,
        actor_heuristic_fraction=actor_heuristic_fraction,
        actor_heuristic_start_updates=actor_heuristic_start_updates,
        actor_heuristic_end_updates=actor_heuristic_end_updates,
        actor_heuristic_final_fraction=actor_heuristic_final_fraction,
        train_on_heuristic_actor_rows=train_on_heuristic_actor_rows,
        diverse_opponent_actor_count=diverse_opponent_actor_count,
        diverse_model_actor_count=diverse_model_actor_count,
        diverse_opponent_batch_fraction=diverse_opponent_batch_fraction,
        diverse_opponent_batch_wait_ms=diverse_opponent_batch_wait_ms,
        heuristic_actor_hidden_state_tracking=heuristic_actor_hidden_state_tracking,
    )
