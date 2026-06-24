"""Small registry adapters used by final policy-set selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from weiss_rl.eval.policies.training_policy_ids import (
    TrainingPolicyId,
    try_parse_training_policy,
    try_parse_training_policy_like,
)


class SnapshotEntryLike(Protocol):
    policy_id: str
    update: int


class _SnapshotRegistryAccess(Protocol):
    snapshots: Sequence[SnapshotEntryLike | str]
    champion_snapshots: Sequence[str]


def snapshot_training_policies(snapshot_registry: object) -> list[TrainingPolicyId]:
    parsed: list[TrainingPolicyId] = []
    for snapshot in snapshot_entries(snapshot_registry):
        candidate = parse_registry_snapshot(snapshot)
        if candidate is not None:
            parsed.append(candidate)
    return parsed


def snapshot_entries(snapshot_registry: object) -> Sequence[SnapshotEntryLike | str]:
    if not hasattr(snapshot_registry, "snapshots"):
        raise TypeError("snapshot_registry must expose a snapshots sequence")
    registry = cast(_SnapshotRegistryAccess, snapshot_registry)
    return registry.snapshots


def champion_snapshot_ids(snapshot_registry: object) -> Sequence[str]:
    if not hasattr(snapshot_registry, "champion_snapshots"):
        raise TypeError("snapshot_registry must expose champion_snapshots")
    registry = cast(_SnapshotRegistryAccess, snapshot_registry)
    return registry.champion_snapshots


def parse_registry_snapshot(snapshot: object) -> TrainingPolicyId | None:
    if isinstance(snapshot, str):
        return try_parse_training_policy(snapshot)
    if hasattr(snapshot, "policy_id") and hasattr(snapshot, "update"):
        snapshot_entry = cast(SnapshotEntryLike, snapshot)
        return try_parse_training_policy_like(
            str(snapshot_entry.policy_id),
            update=int(snapshot_entry.update),
        )
    raise TypeError(f"unsupported snapshot entry type: {type(snapshot).__name__}")


__all__ = [
    "SnapshotEntryLike",
    "champion_snapshot_ids",
    "parse_registry_snapshot",
    "snapshot_entries",
    "snapshot_training_policies",
]
