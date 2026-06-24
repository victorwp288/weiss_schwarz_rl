from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.eval import resolve_final_policy_set
from weiss_rl.eval.final.policy_selection import (
    FINAL_EVAL_POLICY_SELECTION_PATHS,
    load_dev_eval_summaries,
    resolve_final_eval_policy_ids,
    resolve_final_policy_set_with_trace,
    validate_final_eval_seed_budget,
)
from weiss_rl.eval.policies.dev_eval_summaries import DevEvalPolicySummary

from .final_eval_test_support import (
    _selection_config,
    _write_policy_set_inputs,
)


def test_resolve_final_policy_set_uses_deterministic_order_from_artifacts(tmp_path: Path) -> None:
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    config = _selection_config(
        include_random_legal_baseline_b0=True,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=True,
        include_spaced_snapshots_near_percent_updates=(50,),
    )

    selected = resolve_final_policy_set(
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        config=config,
        final_policy_set_size=4,
    )

    assert selected == [
        "B0 RandomLegal",
        "policy_000300",
        "policy_000100",
        "policy_000150",
    ]


def test_final_eval_policy_selection_helpers_preserve_payload_shapes(tmp_path: Path) -> None:
    summaries_path = tmp_path / "dev_eval_summaries.json"
    summaries_path.write_text(
        json.dumps(
            {
                "policy_scalar": 0.25,
                "policy_structured": {
                    "aggregate_score": 0.75,
                    "anchor_scores": {"B0 RandomLegal": 0.9},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = load_dev_eval_summaries(summaries_path)
    resolved, selection_payload = resolve_final_eval_policy_ids(
        policy_ids=["B0 RandomLegal", 123],
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        selection_config=None,
        final_policy_set_size=None,
    )
    validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=1, max_paired_seeds=2)

    assert summaries["policy_scalar"] == 0.25
    assert isinstance(summaries["policy_structured"], DevEvalPolicySummary)
    assert summaries["policy_structured"].anchor_scores == {"B0 RandomLegal": 0.9}
    assert resolved == ["B0 RandomLegal", "123"]
    assert selection_payload == {
        "mode": "explicit",
        "policy_count": 2,
        "selection_trace": [
            {"policy_id": "B0 RandomLegal", "reason": "explicit_policy_id", "details": {"position": 0}},
            {"policy_id": "123", "reason": "explicit_policy_id", "details": {"position": 1}},
        ],
    }


def test_final_eval_policy_selection_trace_explains_deterministic_sources(tmp_path: Path) -> None:
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    config = _selection_config(
        include_random_legal_baseline_b0=True,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=True,
        include_spaced_snapshots_near_percent_updates=(50,),
    )

    selected, trace = resolve_final_policy_set_with_trace(
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        config=config,
        final_policy_set_size=4,
    )

    assert selected == ["B0 RandomLegal", "policy_000300", "policy_000100", "policy_000150"]
    assert [(entry["policy_id"], entry["reason"]) for entry in trace] == [
        ("B0 RandomLegal", "random_legal_baseline_b0"),
        ("policy_000300", "latest_champion_snapshot"),
        ("policy_000100", "spaced_snapshot_near_percent_update"),
        ("policy_000150", "top_dev_performer_vs_anchor_set"),
    ]
    assert trace[-1]["details"]["mean_anchor_score"] == pytest.approx(0.8)


def test_final_eval_policy_selection_paths_document_inputs() -> None:
    paths = {path.mode: path for path in FINAL_EVAL_POLICY_SELECTION_PATHS}

    assert tuple(paths) == ("explicit", "deterministic_v1")
    assert paths["explicit"].required_inputs == ("policy_ids",)
    assert paths["deterministic_v1"].required_inputs == (
        "snapshot_registry_path",
        "dev_eval_summaries_path",
        "selection_config",
        "final_policy_set_size",
    )


def test_final_eval_policy_selection_helpers_preserve_errors() -> None:
    with pytest.raises(ValueError, match="policy_ids must contain at least one policy"):
        resolve_final_eval_policy_ids(
            policy_ids=[],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=None,
            final_policy_set_size=None,
        )
    with pytest.raises(ValueError, match="policy_ids must be unique"):
        resolve_final_eval_policy_ids(
            policy_ids=["policy_dup", "policy_dup"],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=None,
            final_policy_set_size=None,
        )
    with pytest.raises(ValueError, match="missing: snapshot_registry_path, dev_eval_summaries_path"):
        resolve_final_eval_policy_ids(
            policy_ids=None,
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=_selection_config(),
            final_policy_set_size=2,
        )
    with pytest.raises(ValueError, match="stage1_paired_seeds must be positive"):
        validate_final_eval_seed_budget(paired_seeds=[11], stage1_paired_seeds=0, max_paired_seeds=1)
    with pytest.raises(ValueError, match="max_paired_seeds must be >= stage1_paired_seeds"):
        validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=2, max_paired_seeds=1)
    with pytest.raises(ValueError, match="final eval requires at least 3 paired seeds, found 2"):
        validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=1, max_paired_seeds=3)
