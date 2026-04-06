from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from weiss_rl.config import load_stack_config
from weiss_rl.eval.policy_set import (
    DevEvalPolicySummary,
    parse_training_policy_id,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _selection_config(**overrides: Any):
    stack = load_stack_config(_repo_root() / "configs/rl_stack_locked.yaml")
    assert stack.config.evaluation is not None
    return replace(stack.config.evaluation.final_policy_set_selection, **overrides)


def _build_registry(
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


def test_parse_training_policy_id_matches_legacy_repo_snapshot_format() -> None:
    parsed = parse_training_policy_id("train_u50000_p3")

    assert parsed.policy_id == "train_u50000_p3"
    assert parsed.update == 50000
    assert parsed.version == 3


def test_selector_picks_spaced_snapshots_from_durable_registry_updates() -> None:
    config = _selection_config(
        include_random_legal_baseline_b0=False,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
    )
    registry = _build_registry(
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
    config = _selection_config(
        include_heuristic_public_b2_if_exists=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = _build_registry(
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
    config = _selection_config(
        include_heuristic_public_b2_if_exists=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = _build_registry([("policy_000100", 100), ("policy_000200", 200)])
    registry.champion_snapshots = ["policy_999999", "policy_000100"]

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries={},
        config=config,
        final_policy_set_size=4,
    )

    assert selected == ["B0 RandomLegal", "B1 NoLeague baseline", "policy_000100"]


def test_selector_ranks_remaining_slots_by_anchor_set_then_policy_id() -> None:
    config = _selection_config(
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    summaries = {
        "B2 HeuristicPublic": DevEvalPolicySummary(
            policy_id="B2 HeuristicPublic",
            aggregate_score=0.0,
            anchor_scores={},
        ),
        "policy_000150": DevEvalPolicySummary(
            policy_id="policy_000150",
            aggregate_score=0.99,
            anchor_scores={
                "B0 RandomLegal": 0.70,
                "B1 NoLeague baseline": 0.70,
                "B2 HeuristicPublic": 0.70,
            },
        ),
        "policy_000200": DevEvalPolicySummary(
            policy_id="policy_000200",
            aggregate_score=0.10,
            anchor_scores={
                "B0 RandomLegal": 0.70,
                "B1 NoLeague baseline": 0.70,
                "B2 HeuristicPublic": 0.70,
            },
        ),
        "policy_000250": DevEvalPolicySummary(
            policy_id="policy_000250",
            aggregate_score=0.95,
            anchor_scores={
                "B0 RandomLegal": 0.68,
                "B1 NoLeague baseline": 0.67,
                "B2 HeuristicPublic": 0.66,
            },
        ),
    }

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=SnapshotRegistry(),
        dev_eval_summaries=summaries,
        config=config,
        final_policy_set_size=6,
    )

    assert selected == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000150",
        "policy_000200",
        "policy_000250",
    ]


def test_selector_requires_required_anchor_scores_for_anchor_strategy() -> None:
    config = _selection_config(
        include_random_legal_baseline_b0=False,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    summaries = {
        "policy_000150": DevEvalPolicySummary(
            policy_id="policy_000150",
            aggregate_score=0.8,
            anchor_scores={"B0 RandomLegal": 0.8},
        )
    }

    with pytest.raises(ValueError, match="B1 NoLeague baseline"):
        select_final_policy_set_deterministic_v1(
            snapshot_registry=SnapshotRegistry(),
            dev_eval_summaries=summaries,
            config=config,
            final_policy_set_size=1,
        )
