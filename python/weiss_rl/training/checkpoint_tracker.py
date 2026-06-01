"""Checkpoint tracker persistence and schema defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"
CHECKPOINT_TRACKER_FORMAT = "checkpoint_tracker_v1"


class CheckpointTrainingPaths(Protocol):
    @property
    def checkpoint_tracker_path(self) -> Path: ...


def default_checkpoint_tracker_payload() -> dict[str, Any]:
    return {"format": CHECKPOINT_TRACKER_FORMAT, "latest": None, "best": None, "observed_best": None}


def load_checkpoint_tracker(training_paths: CheckpointTrainingPaths) -> dict[str, Any]:
    tracker_path = training_paths.checkpoint_tracker_path
    if not tracker_path.is_file():
        return default_checkpoint_tracker_payload()
    payload = json.loads(tracker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint tracker must be a JSON object: {tracker_path}")
    payload.setdefault("format", CHECKPOINT_TRACKER_FORMAT)
    payload.setdefault("latest", None)
    payload.setdefault("best", None)
    payload.setdefault("observed_best", None)
    return payload


def write_checkpoint_tracker(training_paths: CheckpointTrainingPaths, payload: dict[str, Any]) -> None:
    training_paths.checkpoint_tracker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def best_checkpoint_record(training_paths: CheckpointTrainingPaths) -> Mapping[str, Any] | None:
    best_record = load_checkpoint_tracker(training_paths).get("best")
    return best_record if isinstance(best_record, Mapping) else None


__all__ = [
    "CHECKPOINT_TRACKER_FILENAME",
    "CHECKPOINT_TRACKER_FORMAT",
    "CheckpointTrainingPaths",
    "best_checkpoint_record",
    "default_checkpoint_tracker_payload",
    "load_checkpoint_tracker",
    "write_checkpoint_tracker",
]
