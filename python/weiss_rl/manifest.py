"""Run manifest schemas and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from weiss_rl.repro import sha256_hex


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
    seed_files: dict[str, SeedFileManifest] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    evaluation_pinning: dict[str, Any] = field(default_factory=dict)
    policy_set_selection: list[str] = field(default_factory=list)

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
    manifest_path: Path
    spec_bundle_path: Path
    spec_hash_path: Path
    config_hash_path: Path
    config_json_path: Path


def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def build_seed_file_manifest(seed_files: dict[str, Path], *, root: Path) -> dict[str, SeedFileManifest]:
    return {
        key: SeedFileManifest(path=path.relative_to(root).as_posix(), sha256=sha256_file(path))
        for key, path in sorted(seed_files.items())
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_run_artifacts(base_dir: Path, manifest: RunManifest, *, run_dir_name: str | None = None) -> RunArtifacts:
    directory_name = run_dir_name or f"run_{manifest.run_id64}"
    run_dir = base_dir / directory_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for child in ("checkpoints", "eval", "figures", "logs", "replays"):
        (run_dir / child).mkdir(exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    spec_bundle_path = run_dir / "spec_bundle.json"
    spec_hash_path = run_dir / "spec_hash256.txt"
    config_hash_path = run_dir / "config_hash256.txt"
    config_json_path = run_dir / "config_canonical.json"

    manifest.write_json(manifest_path)
    _write_json(spec_bundle_path, manifest.spec_bundle)
    _write_json(config_json_path, manifest.config_canonical)
    spec_hash_path.write_text(f"{manifest.spec_hash256}\n", encoding="utf-8")
    config_hash_path.write_text(f"{manifest.config_hash256}\n", encoding="utf-8")

    return RunArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        spec_bundle_path=spec_bundle_path,
        spec_hash_path=spec_hash_path,
        config_hash_path=config_hash_path,
        config_json_path=config_json_path,
    )
