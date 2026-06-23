"""Checkpoint metadata observation helpers for actor workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CHECKPOINT_METADATA_STEM = re.compile(r"(?:checkpoint_metadata|checkpoint)_(\d+)")


@dataclass(frozen=True)
class ActorCheckpointMetadataObservation:
    update_count: int
    path: Path


def checkpoint_update_from_path(checkpoint_path: Path) -> int | None:
    match = _CHECKPOINT_METADATA_STEM.fullmatch(checkpoint_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def latest_checkpoint_metadata_update(checkpoint_dir: Path | None) -> int:
    if checkpoint_dir is None:
        return 0

    latest_checkpoint_update = 0
    for pattern in ("checkpoint_metadata_*.json", "checkpoint_*.pt"):
        for checkpoint_path in checkpoint_dir.glob(pattern):
            checkpoint_update = checkpoint_update_from_path(checkpoint_path)
            if checkpoint_update is None:
                continue
            latest_checkpoint_update = max(latest_checkpoint_update, checkpoint_update)
    return latest_checkpoint_update


def checkpoint_metadata_path_for_update(checkpoint_dir: Path | None, update_count: int) -> Path | None:
    if checkpoint_dir is None:
        return None

    for path in (
        checkpoint_dir / f"checkpoint_metadata_{update_count}.json",
        checkpoint_dir / f"checkpoint_{update_count}.pt",
    ):
        if path.exists():
            return path
    return None


def observe_new_checkpoint_metadata(
    checkpoint_dir: Path | None,
    *,
    last_observed_update: int,
) -> ActorCheckpointMetadataObservation | None:
    latest_checkpoint_update = latest_checkpoint_metadata_update(checkpoint_dir)
    if latest_checkpoint_update <= last_observed_update:
        return None

    checkpoint_metadata_path = checkpoint_metadata_path_for_update(checkpoint_dir, latest_checkpoint_update)
    if checkpoint_metadata_path is None:
        return None

    return ActorCheckpointMetadataObservation(
        update_count=latest_checkpoint_update,
        path=checkpoint_metadata_path,
    )


__all__ = [
    "ActorCheckpointMetadataObservation",
    "checkpoint_metadata_path_for_update",
    "checkpoint_update_from_path",
    "latest_checkpoint_metadata_update",
    "observe_new_checkpoint_metadata",
]
