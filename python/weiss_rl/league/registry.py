"""Snapshot registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SnapshotRegistry:
    snapshots: list[str] = field(default_factory=list)


    def add(self, snapshot_id: str) -> None:
        self.snapshots.append(snapshot_id)


    def latest(self, n: int = 1) -> list[str]:
        return self.snapshots[-n:]
