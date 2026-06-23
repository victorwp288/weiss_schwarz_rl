from __future__ import annotations

from dataclasses import replace
from typing import Any

from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath

from ._config_paths import canonical_stack_config_path


def selection_config(**overrides: Any):
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.evaluation is not None
    return replace(stack.config.evaluation.final_policy_set_selection, **overrides)


def build_registry(
    snapshot_specs: list[tuple[str, int]],
    *,
    champion_snapshot_ids: list[str] | tuple[str, ...] = (),
) -> SnapshotRegistry:
    registry = SnapshotRegistry()
    for policy_id, update in snapshot_specs:
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=(policy_id * 64)[:64].ljust(64, "0"),
            path=snapshot_weights_relpath(policy_id),
        )
    for policy_id in champion_snapshot_ids:
        registry.add_champion(policy_id)
    return registry
