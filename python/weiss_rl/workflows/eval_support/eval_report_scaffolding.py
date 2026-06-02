from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.workflows.eval_support.eval_report_io import _load_json_object, _write_json


def _load_run_summary_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.run_summary_path.is_file():
        return _load_json_object(layout.run_summary_path, label="run summary")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    return {
        "kind": "run_summary_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "run_dir": layout.run_dir.as_posix(),
        "artifact_roots": {
            "training": layout.relative(layout.training_dir),
            "eval": layout.relative(layout.eval_dir),
            "replays": layout.relative(layout.replays_dir),
            "tensorboard": layout.relative(layout.tensorboard_dir),
            "figures": layout.relative(layout.figures_dir),
        },
        "manifest_path": layout.relative(layout.manifest_path),
        "environment_path": layout.relative(layout.environment_path),
        "determinism_report_path": layout.relative(layout.determinism_report_path),
        "paper_readiness_summary_path": layout.relative(layout.paper_readiness_summary_path),
        "seed_derivation": manifest.get("seed_derivation", {}),
        "runtime_mode": "interpolated_checkpoint",
        "paper_grade": False,
    }


def _load_determinism_report_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.determinism_report_path.is_file():
        return _load_json_object(layout.determinism_report_path, label="determinism report")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    evaluation_pinning = manifest.get("evaluation_pinning", {})
    if not isinstance(evaluation_pinning, dict):
        evaluation_pinning = {}
    seed_derivation = manifest.get("seed_derivation", {})
    if not isinstance(seed_derivation, dict):
        seed_derivation = {}
    seed_files = manifest.get("seed_files", {})
    if not isinstance(seed_files, dict):
        seed_files = {}
    return {
        "kind": "determinism_report_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "policy_selection_mode": "unresolved",
        "evaluation_pinning": evaluation_pinning,
        "seed_derivation": seed_derivation,
        "seed_files": seed_files,
        "device_policy": {
            "learner": "interpolated_checkpoint",
            "evaluation": evaluation_pinning.get("eval_device", "cpu"),
        },
        "replay_verification": {
            "path": layout.relative(layout.replay_verification_json()),
            "status": "pending",
        },
        "canonical_artifact_hashes": {},
    }


def _load_environment_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.environment_path.is_file():
        return _load_json_object(layout.environment_path, label="environment manifest")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    package_names = ("weiss-rl", "weiss-sim", "torch", "numpy", "scipy", "matplotlib")
    return {
        "kind": "environment_manifest_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {name: _safe_package_version(name) for name in package_names},
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _ensure_run_level_report_scaffolding(layout: ArtifactLayout) -> None:
    if not layout.environment_path.is_file():
        _write_json(layout.environment_path, _load_environment_or_default(layout))
    if not layout.run_summary_path.is_file():
        _write_json(layout.run_summary_path, _load_run_summary_or_default(layout))
    if not layout.determinism_report_path.is_file():
        _write_json(layout.determinism_report_path, _load_determinism_report_or_default(layout))
