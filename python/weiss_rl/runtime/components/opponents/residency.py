"""Resident opponent model selection for the queue runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def resident_opponent_policy_ids(
    *,
    candidate_ids: Sequence[str],
    active_assigned_policy_ids: Sequence[str],
    configured_resident_policy_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return policy IDs that must stay loaded for PFSP, fixed lanes, and in-flight actors."""

    return tuple(
        dict.fromkeys(
            [
                *candidate_ids,
                *active_assigned_policy_ids,
                *configured_resident_policy_ids,
            ]
        )
    )


def load_resident_opponent_models(
    *,
    registry: Any,
    resident_policy_ids: Sequence[str],
    load_snapshot_model: Callable[[str], Any],
) -> dict[str, Any]:
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    models: dict[str, Any] = {}
    for policy_id in resident_policy_ids:
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is None:
            continue
        models[str(policy_id)] = load_snapshot_model(snapshot.path)
    return models


__all__ = ["load_resident_opponent_models", "resident_opponent_policy_ids"]
