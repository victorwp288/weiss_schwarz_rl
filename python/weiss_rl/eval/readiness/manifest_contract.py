"""Manifest consistency checks for paper-readiness audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.readiness.fields import (
    compare_json_file_to_manifest as _compare_json_file_to_manifest,
)
from weiss_rl.eval.readiness.fields import (
    compare_text_file_to_manifest as _compare_text_file_to_manifest,
)
from weiss_rl.eval.readiness.fields import (
    load_json_object as _load_json_object,
)
from weiss_rl.eval.readiness.fields import (
    validate_bool_field as _validate_bool_field,
)
from weiss_rl.eval.readiness.fields import (
    validate_existing_file as _validate_existing_file,
)
from weiss_rl.eval.readiness.fields import (
    validate_hex_field as _validate_hex_field,
)
from weiss_rl.eval.readiness.fields import (
    validate_manifest_policy_set_selection as _validate_manifest_policy_set_selection,
)
from weiss_rl.eval.readiness.fields import (
    validate_object_field as _validate_object_field,
)
from weiss_rl.eval.readiness.fields import (
    validate_seed_files_field as _validate_seed_files_field,
)
from weiss_rl.eval.readiness.fields import (
    validate_simulator_manifest as _validate_simulator_manifest,
)


def build_manifest_contract(run_dir: Path) -> dict[str, Any]:
    layout = ArtifactLayout.from_run_dir(run_dir)
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _load_json_object(manifest_path)
    except Exception as exc:
        return {
            "passed": False,
            "manifest_path": manifest_path.as_posix(),
            "fields": {},
            "consistency_checks": {},
            "missing_fields": [],
            "invalid_fields": [],
            "mismatches": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    field_checks = {
        "run_id256": _validate_hex_field(manifest.get("run_id256"), length=64),
        "run_id64": _validate_hex_field(manifest.get("run_id64"), length=16),
        "git_commit": _validate_hex_field(manifest.get("git_commit"), length=40),
        "git_dirty": _validate_bool_field(manifest.get("git_dirty")),
        "spec_hash256": _validate_hex_field(manifest.get("spec_hash256"), length=64),
        "config_hash256": _validate_hex_field(manifest.get("config_hash256"), length=64),
        "simulator": _validate_simulator_manifest(manifest.get("simulator")),
        "spec_bundle": _validate_object_field(manifest.get("spec_bundle"), require_non_empty=True),
        "config_canonical": _validate_object_field(manifest.get("config_canonical"), require_non_empty=True),
        "seed_files": _validate_seed_files_field(manifest.get("seed_files")),
        "hardware": _validate_object_field(manifest.get("hardware"), require_non_empty=True),
        "evaluation_pinning": _validate_object_field(manifest.get("evaluation_pinning"), require_non_empty=True),
        "policy_set_selection": _validate_manifest_policy_set_selection(
            manifest.get("policy_set_selection"),
            details=manifest.get("policy_set_selection_details"),
        ),
    }
    missing_fields = [name for name, result in field_checks.items() if result["reason"] == "missing"]
    invalid_fields = [
        name for name, result in field_checks.items() if not result["passed"] and result["reason"] != "missing"
    ]

    consistency_checks = {
        "spec_bundle_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "spec_bundle.json",
            expected=manifest.get("spec_bundle"),
        ),
        "config_canonical_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "config_canonical.json",
            expected=manifest.get("config_canonical"),
        ),
        "spec_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "spec_hash256.txt",
            expected=manifest.get("spec_hash256"),
        ),
        "config_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "config_hash256.txt",
            expected=manifest.get("config_hash256"),
        ),
        "run_summary_exists": _validate_existing_file(layout.run_summary_path),
        "environment_manifest_exists": _validate_existing_file(layout.environment_path),
        "determinism_report_exists": _validate_existing_file(layout.determinism_report_path),
    }
    mismatches = [name for name, result in consistency_checks.items() if not result["passed"]]
    passed = not missing_fields and not invalid_fields and not mismatches

    return {
        "passed": passed,
        "manifest_path": manifest_path.as_posix(),
        "fields": field_checks,
        "consistency_checks": consistency_checks,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "mismatches": mismatches,
        "message": (
            "manifest satisfies paper-readiness requirements"
            if passed
            else "manifest is missing required fields or has inconsistent companion files"
        ),
    }
