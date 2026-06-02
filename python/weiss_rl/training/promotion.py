from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
)
from weiss_rl.experiments.baselines import (
    find_noleague_baseline_snapshot as _find_noleague_baseline_snapshot,
)
from weiss_rl.league.registry import SnapshotMeta, SnapshotRegistry

PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = NOLEAGUE_BASELINE_NAME
PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID


def slug_policy_id(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    return "_".join(part for part in parts if part)


def promotion_anchor_policy_id_candidates(anchor_name: str) -> tuple[str, ...]:
    if anchor_name == PROMOTION_GATE_RANDOMLEGAL_NAME:
        return (PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,)
    if anchor_name == PROMOTION_GATE_NOLEAGUE_BASELINE_NAME:
        return (PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID, anchor_name)
    if heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
        return (anchor_name,)
    normalized = slug_policy_id(anchor_name)
    if not normalized:
        return ()
    return tuple(dict.fromkeys((normalized, anchor_name)))


def resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
) -> str | None:
    if anchor_name == "Latest champion snapshot":
        champion_ids = registry.latest_champions(1)
        return None if not champion_ids else str(champion_ids[-1])
    if anchor_name == "Previous champion snapshot":
        champion_ids = registry.latest_champions(2)
        return None if len(champion_ids) < 2 else str(champion_ids[-2])
    if anchor_name == "Latest recent snapshot":
        recent_ids = registry.latest_ids(1)
        return None if not recent_ids else str(recent_ids[-1])
    if anchor_name == "Previous recent snapshot":
        recent_ids = registry.latest_ids(2)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])
    return None


def resolve_promotion_anchor_policy_ids(
    *,
    stack: Any,
    registry: SnapshotRegistry,
) -> tuple[dict[str, str], tuple[str, ...]]:
    league = stack.config.league
    if league is None:
        return {}, ()

    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    resolved: dict[str, str] = {}
    missing_required: list[str] = []
    anchor_names = [
        *league.promotion_anchor_set_v1.required,
        *league.promotion_anchor_set_v1.optional_if_available,
    ]
    required_names = set(league.promotion_anchor_set_v1.required)

    for anchor_name in anchor_names:
        policy_id = resolve_symbolic_promotion_anchor_policy_id(anchor_name, registry=registry)
        if policy_id is None:
            candidates = promotion_anchor_policy_id_candidates(anchor_name)
            policy_id = next((candidate for candidate in candidates if candidate in available_policy_ids), None)
        if policy_id is None and anchor_name == PROMOTION_GATE_RANDOMLEGAL_NAME:
            policy_id = PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        if policy_id is None and heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
            policy_id = anchor_name
        if policy_id is not None:
            resolved[anchor_name] = policy_id
            continue
        if anchor_name in required_names:
            missing_required.append(anchor_name)

    return resolved, tuple(missing_required)


def build_heuristic_public_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
    policy_cls: Any | None = None,
) -> HeuristicPublicPolicy:
    if policy_cls is None:
        policy_cls = HeuristicPublicPolicy
    factory = policy_cls.from_spec_bundle
    supports_scoring_profile = False
    try:
        supports_scoring_profile = "scoring_profile" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        supports_scoring_profile = False
    if supports_scoring_profile:
        return factory(spec_bundle, scoring_profile=scoring_profile)
    return factory(spec_bundle)


def snapshot_meta_by_policy_id(registry: SnapshotRegistry) -> dict[str, SnapshotMeta]:
    return {snapshot.policy_id: snapshot for snapshot in registry.snapshots}


def find_noleague_baseline_snapshot(run_dir: Path) -> SnapshotMeta | None:
    return _find_noleague_baseline_snapshot(
        run_dir,
        policy_id_candidates=promotion_anchor_policy_id_candidates(PROMOTION_GATE_NOLEAGUE_BASELINE_NAME),
    )
