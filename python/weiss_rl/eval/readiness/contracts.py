"""Compatibility surface for paper-readiness artifact contracts."""

from __future__ import annotations

from weiss_rl.eval.readiness.final_eval_contract import (
    build_final_eval_artifact_contract,
    validate_final_eval_policy_set,
)
from weiss_rl.eval.readiness.manifest_contract import build_manifest_contract
from weiss_rl.eval.readiness.run_directory_contract import (
    build_run_directory_audit,
    evaluate_required_artifact,
)
from weiss_rl.eval.readiness.sensitivity_contract import (
    resolve_sensitivity_summary_path,
    sensitivity_root_candidates,
    validate_sensitivity_summary,
)
from weiss_rl.eval.readiness.specs import (
    REQUIRED_ARTIFACT_GROUPS,
    REQUIRED_SENSITIVITY_CASE_IDS,
    RequiredArtifactGroup,
    RequiredArtifactSpec,
    required_run_artifact_group_payload,
    required_run_artifact_specs,
)

__all__ = [
    "REQUIRED_ARTIFACT_GROUPS",
    "REQUIRED_SENSITIVITY_CASE_IDS",
    "RequiredArtifactGroup",
    "RequiredArtifactSpec",
    "build_final_eval_artifact_contract",
    "build_manifest_contract",
    "build_run_directory_audit",
    "evaluate_required_artifact",
    "required_run_artifact_group_payload",
    "required_run_artifact_specs",
    "resolve_sensitivity_summary_path",
    "sensitivity_root_candidates",
    "validate_final_eval_policy_set",
    "validate_sensitivity_summary",
]
