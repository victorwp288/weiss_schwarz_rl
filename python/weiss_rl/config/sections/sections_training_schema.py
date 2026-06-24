"""Allowed keys and value choices for the training config section."""

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

TRAINING_KEYS = frozenset(
    {
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
    }
)
TRAINING_ROLLOUT_KEYS = frozenset({"unroll_length", "batch_unrolls_per_update"})
TRAINING_OPTIMIZER_KEYS = frozenset({"name", "learning_rate", "grad_norm_clip", "value_loss_coef"})
TRAINING_EXPLORATION_KEYS = frozenset(
    {
        "entropy_coef",
        "entropy_anneal_to",
        "entropy_anneal_steps_updates",
        "entropy_scope",
        "actor_sampling_temperature",
    }
)
TRAINING_PRECISION_KEYS = frozenset(
    {"mixed_precision", "compile_learner", "compile_actor_inference", "masking_math_float32"}
)
TRAINING_CHECKPOINTING_KEYS = frozenset(
    {"checkpoint_interval_updates", "snapshot_interval_updates", "actor_reload_interval_updates"}
)
TRAINING_VTRACE_KEYS = frozenset({"rho_bar", "c_bar"})
TRAINING_PPO_KEYS = frozenset(
    {"clip_epsilon", "value_clip_epsilon", "gae_lambda", "epochs", "target_kl", "normalize_advantages"}
)
TRAINING_STRUCTURED_AUX_KEYS = frozenset(
    {
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
    }
)
TRAINING_STRUCTURED_WARMSTART_KEYS = frozenset(
    {
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
    }
)
TRAINING_STRUCTURED_METRICS_KEYS = frozenset({"mode"})
TRAINING_TEACHER_AUX_KEYS = frozenset({"mode"})
TRAINING_ACTION_SURFACE_KEYS = frozenset(
    {
        "mulligan_force_confirm_after_select",
        "force_pass_over_main_move_only",
        "main_move_only_max_consecutive",
        "force_attack_over_pass_when_attack_legal",
    }
)


__all__ = [
    "TRAINING_ACTION_SURFACE_KEYS",
    "TRAINING_ACTOR_POLICY_BACKENDS",
    "TRAINING_ALGORITHMS",
    "TRAINING_CHECKPOINTING_KEYS",
    "TRAINING_ENTROPY_SCOPES",
    "TRAINING_EXPLORATION_KEYS",
    "TRAINING_FIXED_MODEL_OPPONENT_ACTION_SELECTIONS",
    "TRAINING_FIXED_OPPONENT_BACKENDS",
    "TRAINING_KEYS",
    "TRAINING_OPTIMIZER_KEYS",
    "TRAINING_PAIRED_OUTCOME_PREFERENCE_AGGREGATIONS",
    "TRAINING_PAIRED_SWING_COMPARE_TO",
    "TRAINING_PAIRED_SWING_CONFLICT_FILTERS",
    "TRAINING_PAIRED_SWING_LOSS_SCOPES",
    "TRAINING_PPO_KEYS",
    "TRAINING_PRECISION_KEYS",
    "TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES",
    "TRAINING_PUBLIC_HEURISTIC_PROFILES",
    "TRAINING_ROLLOUT_KEYS",
    "TRAINING_STRUCTURED_AUX_KEYS",
    "TRAINING_STRUCTURED_METRICS_KEYS",
    "TRAINING_STRUCTURED_METRICS_MODES",
    "TRAINING_STRUCTURED_WARMSTART_KEYS",
    "TRAINING_TEACHER_AUX_KEYS",
    "TRAINING_TEACHER_AUX_MODES",
    "TRAINING_TRAJECTORY_RETENTION_SOURCES",
    "TRAINING_VTRACE_KEYS",
]
