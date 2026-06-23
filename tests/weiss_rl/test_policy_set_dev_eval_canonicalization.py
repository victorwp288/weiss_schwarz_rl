from __future__ import annotations

from weiss_rl.eval.policies.set import (
    DevEvalPolicySummary,
    select_final_policy_set_deterministic_v1,
)

from .policy_set_test_support import build_registry, selection_config


def test_selector_maps_legacy_dev_eval_policy_ids_to_durable_registry_ids() -> None:
    config = selection_config(
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = build_registry(
        [
            ("policy_000002", 40),
            ("policy_000003", 60),
        ]
    )
    summaries = {
        "B2 HeuristicPublic": DevEvalPolicySummary(
            policy_id="B2 HeuristicPublic",
            aggregate_score=0.0,
            anchor_scores={},
        ),
        "train_u40_p2": DevEvalPolicySummary(
            policy_id="train_u40_p2",
            aggregate_score=0.91,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.75,
                "B2 HeuristicPublic": 1.0,
            },
        ),
        "train_u60_p3": DevEvalPolicySummary(
            policy_id="train_u60_p3",
            aggregate_score=0.92,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.8,
                "B2 HeuristicPublic": 1.0,
            },
        ),
    }

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries=summaries,
        config=config,
        final_policy_set_size=5,
    )

    assert selected == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000003",
        "policy_000002",
    ]


def test_selector_drops_unmapped_legacy_dev_eval_policy_ids() -> None:
    config = selection_config(
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )
    registry = build_registry([("policy_000003", 60)])
    summaries = {
        "B2 HeuristicPublic": DevEvalPolicySummary(
            policy_id="B2 HeuristicPublic",
            aggregate_score=0.0,
            anchor_scores={},
        ),
        "train_u35_p1": DevEvalPolicySummary(
            policy_id="train_u35_p1",
            aggregate_score=0.99,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.8,
                "B2 HeuristicPublic": 1.0,
            },
        ),
        "train_u60_p3": DevEvalPolicySummary(
            policy_id="train_u60_p3",
            aggregate_score=0.92,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.75,
                "B2 HeuristicPublic": 1.0,
            },
        ),
    }

    selected = select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries=summaries,
        config=config,
        final_policy_set_size=4,
    )

    assert selected == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000003",
    ]
