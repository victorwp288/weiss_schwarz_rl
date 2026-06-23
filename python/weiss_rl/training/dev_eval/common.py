"""Shared paths and JSON helpers for periodic dev-eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class DevEvalTrainingPaths(Protocol):
    logs_dir: Path


def resolve_repo_path(root: Path, path_text: str) -> Path:
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else root / candidate


def json_relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def periodic_dev_eval_summaries_path(training_paths: DevEvalTrainingPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_summaries.json"


def stall_monitor_state_path(training_paths: DevEvalTrainingPaths) -> Path:
    return training_paths.logs_dir / "stall_monitor.json"


__all__ = [
    "DevEvalTrainingPaths",
    "json_relative_path",
    "load_json_object",
    "periodic_dev_eval_summaries_path",
    "resolve_repo_path",
    "stall_monitor_state_path",
    "write_json",
]
