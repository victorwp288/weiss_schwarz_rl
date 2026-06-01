"""Core legacy aliases for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Mapping

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
