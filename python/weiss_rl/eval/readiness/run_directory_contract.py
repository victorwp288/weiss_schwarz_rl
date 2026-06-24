"""Run-directory artifact inventory checks for paper-readiness audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.eval.readiness.specs import (
    RequiredArtifactSpec,
    required_run_artifact_group_payload,
    required_run_artifact_specs,
)


def build_run_directory_audit(run_dir: Path) -> dict[str, Any]:
    specs = required_run_artifact_specs()
    artifact_results = {
        spec.artifact_id: evaluate_required_artifact(run_dir=run_dir, spec=spec)
        for spec in specs
    }
    missing_artifacts = [artifact_id for artifact_id, result in artifact_results.items() if not bool(result["passed"])]
    return {
        "passed": not missing_artifacts,
        "artifact_count": len(artifact_results),
        "artifact_groups": required_run_artifact_group_payload(specs),
        "missing_artifacts": missing_artifacts,
        "artifacts": artifact_results,
        "message": (
            "all required run-directory artifacts are present"
            if not missing_artifacts
            else f"missing {len(missing_artifacts)} required artifact checks"
        ),
    }


def evaluate_required_artifact(*, run_dir: Path, spec: RequiredArtifactSpec) -> dict[str, Any]:
    if spec.glob is not None:
        matches = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.glob(spec.glob) if path.is_file())
        passed = len(matches) >= spec.minimum_count
        return {
            "passed": passed,
            "category": spec.category,
            "description": spec.description,
            "glob": spec.glob,
            "minimum_count": spec.minimum_count,
            "matches": matches,
        }

    candidates = [path.as_posix() for path in spec.paths]
    for candidate in spec.paths:
        resolved = run_dir / candidate
        if resolved.is_file():
            return {
                "passed": True,
                "category": spec.category,
                "description": spec.description,
                "expected_paths": candidates,
                "resolved_path": candidate.as_posix(),
            }
    return {
        "passed": False,
        "category": spec.category,
        "description": spec.description,
        "expected_paths": candidates,
        "resolved_path": None,
    }
