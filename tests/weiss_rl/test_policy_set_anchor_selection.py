from __future__ import annotations

from dataclasses import replace

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.eval.policies.set import (
    DevEvalPolicySummary,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.league.registry import SnapshotRegistry

from ._config_paths import repo_root
from .policy_set_test_support import selection_config


def test_selector_ranks_remaining_slots_by_anchor_set_then_policy_id() -> None:
    config = selection_config(
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


def test_thesis_final_eval_selector_includes_b0_through_b4_without_dev_eval_anchor_keys() -> None:
    stack = load_stack_config(repo_root() / "configs" / "thesis" / "final_eval.yaml")
    assert stack.config.evaluation is not None
    config = replace(
        stack.config.evaluation.final_policy_set_selection,
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    summaries = {
        "policy_000150": DevEvalPolicySummary(
            policy_id="policy_000150",
            aggregate_score=0.99,
            anchor_scores={
                "B0 RandomLegal": 0.70,
                "B1 NoLeague baseline": 0.70,
                "B2 HeuristicPublic": 0.70,
                "B3 HeuristicPublicAggro": 0.68,
                "B4 HeuristicPublicControl": 0.67,
            },
        )
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
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
        "policy_000150",
    ]


def test_selector_requires_required_anchor_scores_for_anchor_strategy() -> None:
    config = selection_config(
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
