from __future__ import annotations

from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath


def build_registry(snapshot_ids: list[str], *, champion_snapshot_ids: list[str]) -> SnapshotRegistry:
    registry = SnapshotRegistry()
    for update, snapshot_id in enumerate(snapshot_ids, start=1):
        registry.add_snapshot(
            policy_id=snapshot_id,
            update=update,
            weights_sha256=(snapshot_id * 64)[:64].ljust(64, "0"),
            path=snapshot_weights_relpath(snapshot_id),
        )
    for snapshot_id in champion_snapshot_ids:
        registry.add_champion(snapshot_id)
    return registry
