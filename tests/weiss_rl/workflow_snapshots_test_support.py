from __future__ import annotations

import json
from pathlib import Path


def write_registry(run_dir: Path, payload: object) -> Path:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return registry_path


def write_checkpoint(run_dir: Path, update: int) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    return checkpoint_path
