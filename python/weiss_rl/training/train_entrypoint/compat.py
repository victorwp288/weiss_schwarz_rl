"""Compatibility export and alias installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.core_exports import CORE_COMPAT_EXPORTS
from weiss_rl.training.train_entrypoint.eval_exports import EVAL_COMPAT_EXPORTS
from weiss_rl.training.train_entrypoint.training_exports import TRAINING_COMPAT_EXPORTS

CORE_NAMESPACE_ALIASES: Mapping[str, str] = {
    "_apply_training_flag_overrides": "apply_training_flag_overrides",
    "_expected_sha256": "expected_sha256",
    "_manifest_scaffold_only_reason": "manifest_scaffold_only_reason",
    "_normalize_sha256": "normalize_sha256",
    "_print_manifest_only_message": "print_manifest_only_message",
    "_raise_noleague_training_prerequisite_failure": "raise_noleague_training_prerequisite_failure",
    "_raise_runtime_prerequisite_failure": "raise_runtime_prerequisite_failure",
    "_require_matching_hash": "require_matching_hash",
    "_require_positive_int": "require_positive_int",
    "_resolve_device": "resolve_device",
    "_resolve_run_label": "resolve_run_label",
    "_resolve_runtime_profile": "resolve_runtime_profile",
    "_resolve_seed": "resolve_seed",
    "_runtime_training_prerequisite_failure": "runtime_training_prerequisite_failure",
    "_noleague_training_prerequisite_failure": "noleague_training_prerequisite_failure",
    "_spec_mismatch_policy": "spec_mismatch_policy",
}

EVAL_NAMESPACE_ALIASES: Mapping[str, str] = {
    "DecisionBoundaryBatch": "_DecisionBoundaryBatch",
    "ScheduledGame": "_ScheduledGame",
    "_build_env": "build_training_env",
    "_build_ids_eval_env": "build_ids_eval_env",
    "_env_pool_config": "env_pool_config",
    "_load_dev_eval_summaries": "load_dev_eval_summaries",
    "_policy_set_selection": "policy_set_selection",
    "_resolve_policy_set_selection": "resolve_policy_set_selection",
    "_selection_requires_dev_eval_summaries": "selection_requires_dev_eval_summaries",
    "_selection_requires_snapshot_registry": "selection_requires_snapshot_registry",
    "_spec_dimensions": "spec_dimensions",
    "_json_relative_path": "json_relative_path",
    "_write_json": "write_json",
    "_slug_policy_id": "slug_policy_id",
    "_promotion_anchor_policy_id_candidates": "promotion_anchor_policy_id_candidates",
    "_resolve_symbolic_promotion_anchor_policy_id": "resolve_symbolic_promotion_anchor_policy_id",
    "_find_noleague_baseline_snapshot": "find_noleague_baseline_snapshot",
    "_resolve_promotion_anchor_policy_ids": "resolve_promotion_anchor_policy_ids",
    "_snapshot_meta_by_policy_id": "snapshot_meta_by_policy_id",
    "_evaluation_config_or_raise": "evaluation_config_or_raise",
    "_validate_periodic_dev_eval_contract": "validate_periodic_dev_eval_contract",
    "_resolve_repo_path": "resolve_repo_path",
    "_resolve_periodic_dev_eval_seed_file": "resolve_periodic_dev_eval_seed_file",
    "_periodic_dev_eval_schedule": "periodic_dev_eval_schedule",
    "_legal_ids_for_env_row": "legal_ids_for_env_row",
    "_periodic_dev_eval_rng_seed": "periodic_dev_eval_rng_seed",
    "_promotion_gate_rng_seed": "promotion_gate_rng_seed",
    "_periodic_dev_eval_bootstrap_seed": "periodic_dev_eval_bootstrap_seed",
    "_promotion_gate_bootstrap_seed": "promotion_gate_bootstrap_seed",
    "_clone_cpu_eval_model": "clone_cpu_eval_model",
    "_current_focal_policy_id": "current_focal_policy_id",
    "_checkpoint_path_for_update": "checkpoint_path_for_update",
    "_should_run_periodic_dev_eval": "should_run_periodic_dev_eval",
    "_periodic_dev_eval_summaries_path": "periodic_dev_eval_summaries_path",
    "_stall_monitor_state_path": "stall_monitor_state_path",
    "_persist_periodic_dev_eval_summary": "persist_periodic_dev_eval_summary",
    "_load_snapshot_registry": "load_snapshot_registry",
}

TRAINING_NAMESPACE_ALIASES: Mapping[str, str] = {
    "MinimalRollout": "_TrainingMinimalRollout",
    "_LATEST_CHECKPOINT_FILENAME": "LATEST_CHECKPOINT_FILENAME",
    "_BEST_CHECKPOINT_FILENAME": "BEST_CHECKPOINT_FILENAME",
    "_CHECKPOINT_TRACKER_FILENAME": "CHECKPOINT_TRACKER_FILENAME",
    "_IMPALA_ALGORITHMS": "IMPALA_ALGORITHMS",
    "_PPO_ALGORITHMS": "PPO_ALGORITHMS",
    "_append_checkpoint_guard_event": "append_checkpoint_guard_event",
    "_assert_noleague_baseline_config": "assert_noleague_baseline_config",
    "_best_checkpoint_record": "best_checkpoint_record",
    "_bootstrap_values": "bootstrap_values",
    "_build_learner_batch": "build_learner_batch",
    "_canonical_config_sections": "canonical_config_sections",
    "_central_runtime_actor_torch_threads": "central_runtime_actor_torch_threads",
    "_checkpoint_guard_log_path": "checkpoint_guard_log_path",
    "_config_marks_noleague_baseline": "config_marks_noleague_baseline",
    "_configure_torch_threads": "configure_torch_threads",
    "_delete_pruned_snapshot_artifacts": "delete_pruned_snapshot_artifacts",
    "_demote_registry_champions_newer_than": "demote_registry_champions_newer_than",
    "_evaluation_pinning": "evaluation_pinning",
    "_extract_structured_guard_b2_anchor_score": "extract_structured_guard_b2_anchor_score",
    "_is_noleague_baseline_role": "is_noleague_baseline_role",
    "_legacy_noleague_baseline_mode": "legacy_noleague_baseline_mode",
    "_load_checkpoint_tracker": "load_checkpoint_tracker",
    "_load_json_object": "load_json_object",
    "_manifest_actor_device_layout": "manifest_actor_device_layout",
    "_manifest_source_path": "manifest_source_path",
    "_maybe_log_structured_mainmove_guard": "maybe_log_structured_mainmove_guard",
    "_read_optional_hash_file": "read_optional_hash_file",
    "_relative_path_text": "relative_path_text",
    "_resolve_resume_checkpoint_path": "resolve_resume_checkpoint_path",
    "_role_from_config_canonical": "role_from_config_canonical",
    "_run_artifacts_from_existing_run_dir": "run_artifacts_from_existing_run_dir",
    "_save_snapshot_registry_with_retention": "save_snapshot_registry_with_retention",
    "_sha256_file": "sha256_file",
    "_snapshot_artifact_dir_for_prune": "snapshot_artifact_dir_for_prune",
    "_start_nonce": "start_nonce",
    "_sync_snapshot_registry_retention": "sync_snapshot_registry_retention",
    "_torch_num_threads_scope": "torch_num_threads_scope",
    "_training_paths": "_training_paths_impl",
    "_validate_imported_snapshot_contract": "validate_imported_snapshot_contract",
    "_write_checkpoint_tracker": "write_checkpoint_tracker",
    "_write_json_file": "write_json_file",
    "_write_scalars_record": "write_scalars_record",
    "_write_snapshot_artifact": "write_snapshot_artifact",
    "_hardware_summary": "hardware_summary",
    "_entropy_coef_for_next_update": "entropy_coef_for_next_update",
    "_teacher_public_heuristic_coef_for_next_update": "teacher_public_heuristic_coef_for_next_update",
    "_public_heuristic_logit_bias_scale_for_next_update": "public_heuristic_logit_bias_scale_for_next_update",
    "_apply_guidance_schedule_for_next_update": "apply_guidance_schedule_for_next_update",
    "_model_guidance_payload": "model_guidance_payload",
    "_restore_model_guidance_from_payload": "restore_model_guidance_from_payload",
    "_maybe_compile_learner_model": "maybe_compile_learner_model",
    "_validate_algorithm_model_contract": "validate_algorithm_model_contract",
}

CHECKPOINT_GUARD_ALIASES: Mapping[str, str] = {
    "_checkpoint_candidate_metric": "checkpoint_candidate_metric",
    "_confirmatory_dev_eval_request": "confirmatory_dev_eval_request",
    "_confirmatory_dev_eval_target_pairs": "confirmatory_dev_eval_target_pairs",
    "_dev_eval_aggregate_score": "dev_eval_aggregate_score",
    "_dev_eval_confidence_stats": "dev_eval_confidence_stats",
    "_dev_eval_ineligibility_reasons": "dev_eval_ineligibility_reasons",
    "_dev_eval_metric_eligible": "dev_eval_metric_eligible",
    "_dev_eval_worst_natural_timeout_rate": "dev_eval_worst_natural_timeout_rate",
    "_dev_eval_worst_no_progress_timeout_rate": "dev_eval_worst_no_progress_timeout_rate",
    "_dev_eval_worst_reason_rate": "dev_eval_worst_reason_rate",
    "_dev_eval_worst_stall_rate": "dev_eval_worst_stall_rate",
    "_dev_eval_worst_truncation_rate": "dev_eval_worst_truncation_rate",
    "_expand_periodic_dev_eval_paired_seeds": "expand_periodic_dev_eval_paired_seeds",
    "_should_promote_best_checkpoint": "should_promote_best_checkpoint",
    "_summary_rate": "summary_rate",
}

COMPAT_EXPORT_FAMILIES: tuple[Mapping[str, Any], ...] = (
    CORE_COMPAT_EXPORTS,
    TRAINING_COMPAT_EXPORTS,
    EVAL_COMPAT_EXPORTS,
)

NAMESPACE_ALIAS_FAMILIES: tuple[Mapping[str, str], ...] = (
    CORE_NAMESPACE_ALIASES,
    TRAINING_NAMESPACE_ALIASES,
    EVAL_NAMESPACE_ALIASES,
)


def install_train_entrypoint_compat_exports(namespace: MutableMapping[str, Any]) -> None:
    for exports in COMPAT_EXPORT_FAMILIES:
        namespace.update(exports)


def install_train_entrypoint_aliases(
    namespace: MutableMapping[str, Any],
    *,
    checkpoint_guard_helpers: Any,
) -> None:
    for aliases in NAMESPACE_ALIAS_FAMILIES:
        for alias_name, source_name in aliases.items():
            namespace[alias_name] = namespace[source_name]
    for alias_name, helper_name in CHECKPOINT_GUARD_ALIASES.items():
        namespace[alias_name] = getattr(checkpoint_guard_helpers, helper_name)


__all__ = [
    "CHECKPOINT_GUARD_ALIASES",
    "COMPAT_EXPORT_FAMILIES",
    "CORE_NAMESPACE_ALIASES",
    "EVAL_NAMESPACE_ALIASES",
    "NAMESPACE_ALIAS_FAMILIES",
    "TRAINING_NAMESPACE_ALIASES",
    "install_train_entrypoint_aliases",
    "install_train_entrypoint_compat_exports",
]
