from __future__ import annotations

from weiss_rl.eval.policies.set import select_final_policy_set_deterministic_v1
from weiss_rl.league.registry import snapshot_weights_relpath

from .policy_set_test_support import build_registry, selection_config


def test_selector_picks_spaced_snapshots_from_durable_registry_updates() -> None:
    config = selection_config(
        include_random_legal_baseline_b0=False,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
    )
    registry = build_registry(
        [
            ("policy_000001", 10),
            ("policy_000004", 40),
            ("policy_000007", 75),
            ("policy_000010", 100),
        ]
    )

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries={},
        config=config,
        final_policy_set_size=3,
    )

    assert selected == ["policy_000001", "policy_000004", "policy_000007"]


def test_selector_uses_latest_champion_snapshot_not_latest_snapshot() -> None:
    config = selection_config(
        include_heuristic_public_b2_if_exists=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = build_registry(
        [("policy_000100", 100), ("policy_000200", 200)],
        champion_snapshot_ids=["policy_000100"],
    )

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries={},
        config=config,
        final_policy_set_size=4,
    )

    assert selected == ["B0 RandomLegal", "B1 NoLeague baseline", "policy_000100"]


def test_selector_ignores_orphan_champion_refs_when_picking_final_champion() -> None:
    config = selection_config(
        include_heuristic_public_b2_if_exists=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = build_registry([("policy_000100", 100), ("policy_000200", 200)])
    registry.champion_snapshots = ["policy_999999", "policy_000100"]

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries={},
        config=config,
        final_policy_set_size=4,
    )

    assert selected == ["B0 RandomLegal", "B1 NoLeague baseline", "policy_000100"]


def test_selector_ignores_non_training_snapshot_policy_ids_in_registry() -> None:
    config = selection_config(
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(100,),
    )
    registry = build_registry([("policy_000100", 100)])
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256=("b1_noleague_baseline" * 64)[:64].ljust(64, "0"),
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries={},
        config=config,
        final_policy_set_size=3,
    )

    assert selected == ["B0 RandomLegal", "B1 NoLeague baseline", "policy_000100"]
