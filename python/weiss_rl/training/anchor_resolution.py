"""Promotion/dev-eval anchor resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.eval.policy_set import heuristic_public_profile_name_for_policy_id
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.training.eval_schedule import PeriodicDevEvalOpponentSpec

PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
FIXED_OPPONENT_EXCLUSIONS = frozenset({PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID})


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


def true_local_recent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    promotion_gate_enabled: bool = False,
) -> tuple[str, ...]:
    if promotion_gate_enabled:
        return tuple(
            registry.latest_active_champion_ids(
                len(getattr(registry, "champion_snapshots", ())),
                exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
            )
        )
    return tuple(
        registry.latest_local_candidate_ids(
            len(getattr(registry, "snapshots", ())),
            include_league_import=True,
            exclude_rejected=True,
            exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
    )


def resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
    promotion_gate_enabled: bool = False,
) -> str | None:
    if anchor_name in {"Latest champion snapshot", "Latest promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            1,
            exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not champion_ids else str(champion_ids[-1])
    if anchor_name in {"Previous champion snapshot", "Previous promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            2,
            exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(champion_ids) < 2 else str(champion_ids[-2])
    if anchor_name in {"Latest recent snapshot", "Latest local candidate snapshot"}:
        recent_ids = true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if not recent_ids else str(recent_ids[-1])
    if anchor_name in {"Previous recent snapshot", "Previous local candidate snapshot"}:
        recent_ids = true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])
    if anchor_name == "Latest imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            1,
            exclude_rejected=True,
            exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not seed_ids else str(seed_ids[-1])
    if anchor_name == "Previous imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            2,
            exclude_rejected=True,
            exclude_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(seed_ids) < 2 else str(seed_ids[-2])
    return None


def resolve_promotion_anchor_policy_ids(
    *,
    stack: StackConfig,
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
    promotion_gate_enabled = bool(getattr(league, "promotion_gate_enabled", False))

    for anchor_name in anchor_names:
        policy_id = resolve_symbolic_promotion_anchor_policy_id(
            anchor_name,
            registry=registry,
            promotion_gate_enabled=promotion_gate_enabled,
        )
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


def snapshot_meta_by_policy_id(registry: SnapshotRegistry) -> dict[str, Any]:
    return {snapshot.policy_id: snapshot for snapshot in registry.snapshots}


def resolve_periodic_dev_eval_opponent_specs(
    *,
    stack: StackConfig,
    run_dir: Path,
) -> tuple[tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    if league is None:
        anchor_names = [PROMOTION_GATE_RANDOMLEGAL_NAME, PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = snapshot_meta_by_policy_id(registry)
    specs: list[PeriodicDevEvalOpponentSpec] = []
    pinned_snapshot_ids: list[str] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="random_legal",
                )
            )
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="heuristic_public",
                    heuristic_profile=heuristic_profile,
                )
            )
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        specs.append(
            PeriodicDevEvalOpponentSpec(
                policy_id=policy_id,
                display_name=anchor_name,
                kind="snapshot",
                snapshot_path=snapshot.path,
            )
        )
        pinned_snapshot_ids.append(policy_id)
    return tuple(specs), tuple(dict.fromkeys(pinned_snapshot_ids))


def resolve_promotion_gate_anchor_specs(
    *,
    stack: StackConfig,
    snapshots_dir: Path,
) -> tuple[dict[str, str], tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    registry_path = snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Promotion gate is missing required anchors: {missing_text}")

    snapshot_index = snapshot_meta_by_policy_id(registry)
    specs: list[PeriodicDevEvalOpponentSpec] = []
    pinned_snapshot_ids: list[str] = []
    for anchor_name, policy_id in anchor_policy_ids.items():
        if policy_id == PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="random_legal",
                )
            )
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="heuristic_public",
                    heuristic_profile=heuristic_profile,
                )
            )
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            raise RuntimeError(f"Promotion gate could not resolve snapshot anchor for policy_id={policy_id}")
        specs.append(
            PeriodicDevEvalOpponentSpec(
                policy_id=policy_id,
                display_name=anchor_name,
                kind="snapshot",
                snapshot_path=snapshot.path,
            )
        )
        pinned_snapshot_ids.append(policy_id)
    return dict(anchor_policy_ids), tuple(specs), tuple(dict.fromkeys(pinned_snapshot_ids))
