"""Training, checkpoint, and snapshot exports for the entrypoint facade."""

from __future__ import annotations

# ruff: noqa: F401
from collections.abc import Mapping
from typing import Any

from weiss_rl.training.algorithm_contracts import validate_algorithm_model_contract
from weiss_rl.training.algorithm_families import (
    IMPALA_ALGORITHMS,
    PPO_ALGORITHMS,
    STRUCTURED_VTRACE_ALGORITHMS,
    training_algorithm_family,
)
from weiss_rl.training.batches import (
    MinimalRollout as _TrainingMinimalRollout,
)
from weiss_rl.training.batches import (
    bootstrap_values,
    build_learner_batch,
)
from weiss_rl.training.batches import (
    collect_training_batch as collect_training_batch,
)
from weiss_rl.training.checkpointing import guard as _checkpoint_guard_helpers
from weiss_rl.training.checkpointing.resolution import (
    BEST_CHECKPOINT_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    resolve_resume_checkpoint_path,
)
from weiss_rl.training.checkpoints import (
    CHECKPOINT_TRACKER_FILENAME,
    ResumeCheckpoint,
    append_checkpoint_guard_event,
    best_checkpoint_record,
    build_checkpoint_record,
    checkpoint_guard_log_path,
    checkpoint_path_for_update,
    current_focal_policy_id,
    extract_structured_guard_b2_anchor_score,
    initialize_model_from_checkpoint,
    load_checkpoint_tracker,
    maybe_log_structured_mainmove_guard,
    publish_checkpoint_aliases,
    relative_path_text,
    restore_minimal_train_checkpoint,
    write_checkpoint_tracker,
    write_minimal_train_checkpoint,
    write_scalars_record,
)
from weiss_rl.training.checkpoints import (
    ensure_current_checkpoint as ensure_current_checkpoint,
)
from weiss_rl.training.checkpoints import (
    maybe_finalize_from_best_checkpoint as maybe_finalize_from_best_checkpoint,
)
from weiss_rl.training.checkpoints import (
    maybe_rollback_to_best_checkpoint as maybe_rollback_to_best_checkpoint,
)
from weiss_rl.training.guidance import (
    apply_guidance_schedule_for_next_update,
    entropy_coef_for_next_update,
    model_guidance_payload,
    public_heuristic_logit_bias_scale_for_next_update,
    restore_model_guidance_from_payload,
    teacher_public_heuristic_coef_for_next_update,
)
from weiss_rl.training.import_contracts import (
    assert_noleague_baseline_config,
    canonical_config_sections,
    config_marks_noleague_baseline,
    is_noleague_baseline_role,
    legacy_noleague_baseline_mode,
    read_optional_hash_file,
    role_from_config_canonical,
    validate_imported_snapshot_contract,
)
from weiss_rl.training.inputs import (
    expected_sha256,
    normalize_sha256,
    require_matching_hash,
    require_positive_int,
    resolve_run_label,
    spec_mismatch_policy,
)
from weiss_rl.training.learner_compile import maybe_compile_learner_model
from weiss_rl.training.learner_factory import build_training_learner
from weiss_rl.training.manifest_layout import manifest_actor_device_layout
from weiss_rl.training.minimal.loop import MinimalTrainingHooks as MinimalTrainingHooks
from weiss_rl.training.minimal.loop import run_minimal_training as run_minimal_training
from weiss_rl.training.noleague_anchor import ensure_noleague_baseline_anchor
from weiss_rl.training.paths import (
    TrainingPaths,
    run_artifacts_from_existing_run_dir,
)
from weiss_rl.training.paths import (
    training_paths as _training_paths_impl,
)
from weiss_rl.training.profiling import build_training_profiler as build_training_profiler
from weiss_rl.training.profiling import profile_block as profile_block
from weiss_rl.training.run_metadata import (
    evaluation_pinning,
    git_commit,
    git_dirty,
    git_output,
    hardware_summary,
    load_json_object,
    manifest_source_path,
    repo_root,
    start_nonce,
)
from weiss_rl.training.seed_snapshots import (
    import_seed_snapshot_pool,
    validate_seed_snapshot_import_contract,
)
from weiss_rl.training.snapshots import (
    delete_pruned_snapshot_artifacts,
    demote_registry_champions_newer_than,
    persist_snapshot_registry_entry,
    save_snapshot_registry_with_retention,
    seed_snapshot_policy_id,
    sha256_file,
    snapshot_artifact_dir_for_prune,
    sync_snapshot_registry_retention,
    write_imported_snapshot_artifact,
    write_json_file,
    write_snapshot_artifact,
)
from weiss_rl.training.torch_threads import (
    central_runtime_actor_torch_threads,
    configure_torch_threads,
    torch_num_threads_scope,
)
from weiss_rl.training.warmstart import run_structured_warmstart

_TRAINING_EXPORT_NAMES = (
    "_checkpoint_guard_helpers",
    "validate_algorithm_model_contract",
    "IMPALA_ALGORITHMS",
    "PPO_ALGORITHMS",
    "STRUCTURED_VTRACE_ALGORITHMS",
    "training_algorithm_family",
    "bootstrap_values",
    "build_learner_batch",
    "_TrainingMinimalRollout",
    "collect_training_batch",
    "BEST_CHECKPOINT_FILENAME",
    "CHECKPOINT_TRACKER_FILENAME",
    "LATEST_CHECKPOINT_FILENAME",
    "ResumeCheckpoint",
    "append_checkpoint_guard_event",
    "best_checkpoint_record",
    "build_checkpoint_record",
    "checkpoint_guard_log_path",
    "checkpoint_path_for_update",
    "current_focal_policy_id",
    "extract_structured_guard_b2_anchor_score",
    "initialize_model_from_checkpoint",
    "load_checkpoint_tracker",
    "maybe_log_structured_mainmove_guard",
    "publish_checkpoint_aliases",
    "relative_path_text",
    "resolve_resume_checkpoint_path",
    "restore_minimal_train_checkpoint",
    "write_checkpoint_tracker",
    "write_minimal_train_checkpoint",
    "write_scalars_record",
    "ensure_current_checkpoint",
    "maybe_finalize_from_best_checkpoint",
    "maybe_rollback_to_best_checkpoint",
    "apply_guidance_schedule_for_next_update",
    "entropy_coef_for_next_update",
    "model_guidance_payload",
    "public_heuristic_logit_bias_scale_for_next_update",
    "restore_model_guidance_from_payload",
    "teacher_public_heuristic_coef_for_next_update",
    "assert_noleague_baseline_config",
    "canonical_config_sections",
    "config_marks_noleague_baseline",
    "is_noleague_baseline_role",
    "legacy_noleague_baseline_mode",
    "read_optional_hash_file",
    "role_from_config_canonical",
    "validate_imported_snapshot_contract",
    "expected_sha256",
    "normalize_sha256",
    "require_matching_hash",
    "require_positive_int",
    "resolve_run_label",
    "spec_mismatch_policy",
    "maybe_compile_learner_model",
    "build_training_learner",
    "manifest_actor_device_layout",
    "MinimalTrainingHooks",
    "run_minimal_training",
    "ensure_noleague_baseline_anchor",
    "TrainingPaths",
    "run_artifacts_from_existing_run_dir",
    "_training_paths_impl",
    "build_training_profiler",
    "profile_block",
    "evaluation_pinning",
    "git_commit",
    "git_dirty",
    "git_output",
    "hardware_summary",
    "load_json_object",
    "manifest_source_path",
    "repo_root",
    "start_nonce",
    "import_seed_snapshot_pool",
    "validate_seed_snapshot_import_contract",
    "delete_pruned_snapshot_artifacts",
    "demote_registry_champions_newer_than",
    "persist_snapshot_registry_entry",
    "save_snapshot_registry_with_retention",
    "seed_snapshot_policy_id",
    "sha256_file",
    "snapshot_artifact_dir_for_prune",
    "sync_snapshot_registry_retention",
    "write_imported_snapshot_artifact",
    "write_json_file",
    "write_snapshot_artifact",
    "central_runtime_actor_torch_threads",
    "configure_torch_threads",
    "torch_num_threads_scope",
    "run_structured_warmstart",
)

TRAINING_COMPAT_EXPORTS: Mapping[str, Any] = {
    **{name: globals()[name] for name in _TRAINING_EXPORT_NAMES},
    "_checkpoint_guard_helpers": _checkpoint_guard_helpers,
    "_TrainingMinimalRollout": _TrainingMinimalRollout,
    "_training_paths_impl": _training_paths_impl,
    "hardware_summary": hardware_summary,
}

__all__ = ["TRAINING_COMPAT_EXPORTS", *_TRAINING_EXPORT_NAMES]
