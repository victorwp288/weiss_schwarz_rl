"""Run manifest schemas and helpers."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout, default_run_dir_name
from weiss_rl.artifacts.reproducibility import hash_seed_file

ARTIFACT_SCHEMA_VERSION = "run_artifacts_v2"


@dataclass(frozen=True, slots=True)
class SeedFileManifest:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id256: str
    run_id64: str
    start_nonce: int
    git_commit: str
    git_dirty: bool
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spec_hash256: str = ""
    config_hash256: str = ""
    simulator: dict[str, Any] = field(default_factory=dict)
    spec_bundle: dict[str, Any] = field(default_factory=dict)
    config_canonical: dict[str, Any] = field(default_factory=dict)
    seed_derivation: dict[str, Any] = field(default_factory=dict)
    seed_files: dict[str, SeedFileManifest] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    evaluation_pinning: dict[str, Any] = field(default_factory=dict)
    policy_set_selection: list[str] = field(default_factory=list)
    policy_set_selection_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seed_files"] = {key: asdict(value) for key, value in self.seed_files.items()}
        return payload

    def write_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    run_dir: Path
    run_dir_name: str
    layout: ArtifactLayout
    manifest_path: Path
    spec_bundle_path: Path
    spec_hash_path: Path
    config_hash_path: Path
    config_json_path: Path
    environment_path: Path
    run_summary_path: Path
    determinism_report_path: Path
    paper_readiness_summary_path: Path
    performance_log_path: Path


def build_seed_file_manifest(seed_files: dict[str, Path], *, root: Path) -> dict[str, SeedFileManifest]:
    return {
        key: SeedFileManifest(path=path.relative_to(root).as_posix(), sha256=hash_seed_file(path))
        for key, path in sorted(seed_files.items())
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_run_artifacts(base_dir: Path, manifest: RunManifest, *, run_label: str | None = None) -> RunArtifacts:
    directory_name = run_label or default_run_dir_name(manifest.run_id64)
    run_dir = base_dir / directory_name
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    for legacy_dir in ("checkpoints", "logs"):
        (run_dir / legacy_dir).mkdir(parents=True, exist_ok=True)

    manifest.write_json(layout.manifest_path)
    _write_json(layout.spec_bundle_path, manifest.spec_bundle)
    _write_json(layout.config_json_path, manifest.config_canonical)
    layout.spec_hash_path.write_text(f"{manifest.spec_hash256}\n", encoding="utf-8")
    layout.config_hash_path.write_text(f"{manifest.config_hash256}\n", encoding="utf-8")

    _write_json(layout.environment_path, default_environment_payload(manifest=manifest))
    _write_json(layout.run_summary_path, default_run_summary_payload(manifest=manifest, layout=layout))
    _write_json(layout.determinism_report_path, default_determinism_report_payload(manifest=manifest, layout=layout))

    return RunArtifacts(
        run_dir=run_dir,
        run_dir_name=directory_name,
        layout=layout,
        manifest_path=layout.manifest_path,
        spec_bundle_path=layout.spec_bundle_path,
        spec_hash_path=layout.spec_hash_path,
        config_hash_path=layout.config_hash_path,
        config_json_path=layout.config_json_path,
        environment_path=layout.environment_path,
        run_summary_path=layout.run_summary_path,
        determinism_report_path=layout.determinism_report_path,
        paper_readiness_summary_path=layout.paper_readiness_summary_path,
        performance_log_path=layout.performance_log_path,
    )


def update_run_summary(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def update_environment_payload(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def update_determinism_report(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def default_environment_payload(*, manifest: RunManifest) -> dict[str, Any]:
    package_names = ("weiss-rl", "weiss-sim", "torch", "numpy", "scipy", "matplotlib")
    return {
        "kind": "environment_manifest_v1",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id256": manifest.run_id256,
        "run_id64": manifest.run_id64,
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


def default_run_summary_payload(*, manifest: RunManifest, layout: ArtifactLayout) -> dict[str, Any]:
    return {
        "kind": "run_summary_v1",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id256": manifest.run_id256,
        "run_id64": manifest.run_id64,
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
        "seed_derivation": manifest.seed_derivation,
        "runtime_mode": manifest.policy_set_selection_details.get("runtime_mode", "manifest_only"),
        "paper_grade": False,
    }


def default_determinism_report_payload(*, manifest: RunManifest, layout: ArtifactLayout) -> dict[str, Any]:
    return {
        "kind": "determinism_report_v1",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id256": manifest.run_id256,
        "run_id64": manifest.run_id64,
        "policy_selection_mode": manifest.policy_set_selection_details.get("mode", "unresolved"),
        "evaluation_pinning": manifest.evaluation_pinning,
        "seed_derivation": manifest.seed_derivation,
        "seed_files": {key: asdict(value) for key, value in manifest.seed_files.items()},
        "device_policy": {
            "learner": manifest.hardware.get("learner_device", "unknown"),
            "evaluation": manifest.evaluation_pinning.get("eval_device", "cpu"),
        },
        "replay_verification": {
            "path": layout.relative(layout.replay_verification_json()),
            "status": "pending",
        },
        "canonical_artifact_hashes": {},
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None
