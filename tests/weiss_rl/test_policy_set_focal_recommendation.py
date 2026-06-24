from __future__ import annotations

from weiss_rl.eval.policies.dev_eval_summaries import DevEvalPolicySummary
from weiss_rl.eval.policies.set import recommend_focal_policy_id
from weiss_rl.league.registry import SnapshotRegistry

from .policy_set_test_support import build_registry


def test_recommend_focal_policy_id_prefers_best_canonicalized_non_baseline_policy() -> None:
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
            aggregate_score=0.89,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.7,
                "B2 HeuristicPublic": 1.0,
            },
        ),
        "train_u60_p3": DevEvalPolicySummary(
            policy_id="train_u60_p3",
            aggregate_score=0.93,
            anchor_scores={
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.8,
                "B2 HeuristicPublic": 1.0,
            },
        ),
    }

    recommended = recommend_focal_policy_id(
        snapshot_registry=registry,
        dev_eval_summaries=summaries,
        candidate_policy_ids=[
            "B0 RandomLegal",
            "b1_noleague_baseline",
            "B2 HeuristicPublic",
            "policy_000002",
            "policy_000003",
        ],
    )

    assert recommended == "policy_000003"


def test_recommend_focal_policy_id_falls_back_to_newest_durable_snapshot_when_summaries_are_missing() -> None:
    registry = build_registry(
        [
            ("policy_000100", 100),
            ("policy_000200", 200),
        ]
    )

    recommended = recommend_focal_policy_id(
        snapshot_registry=registry,
        dev_eval_summaries={},
        candidate_policy_ids=[
            "B0 RandomLegal",
            "B1 NoLeague baseline",
            "policy_000100",
            "policy_000200",
        ],
    )

    assert recommended == "policy_000200"


def test_recommend_focal_policy_id_falls_back_to_newest_legacy_snapshot_when_summaries_are_missing() -> None:
    recommended = recommend_focal_policy_id(
        snapshot_registry=SnapshotRegistry(),
        dev_eval_summaries={},
        candidate_policy_ids=[
            "B0 RandomLegal",
            "train_u40_p2",
            "train_u60_p3",
        ],
    )

    assert recommended == "train_u60_p3"
