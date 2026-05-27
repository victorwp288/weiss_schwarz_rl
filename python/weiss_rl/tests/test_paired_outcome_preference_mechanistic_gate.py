from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate import (
    PairedOutcomePreferenceMechanisticGateConfig,
    evaluate_paired_outcome_preference_mechanistic_gate,
)


def _write_report(
    path: Path,
    *,
    margins: dict[str, list[float]],
    context_count: int = 0,
    episode_count: int | None = None,
    missing_context_opponents: list[str] | None = None,
) -> Path:
    rows = []
    pair_id = 0
    for group_label, group_margins in margins.items():
        for margin in group_margins:
            rows.append(
                {
                    "preference_pair_id": pair_id,
                    "group_label": group_label,
                    "opponent_policy_id": f"{group_label}_opponent",
                    "source_pair_index": pair_id + 100,
                    "preferred_label": f"{group_label}_preferred",
                    "rejected_label": f"{group_label}_rejected",
                    "dpo_margin": margin,
                    "current_raw_margin": margin + 0.5,
                    "reference_raw_margin": 0.5,
                }
            )
            pair_id += 1
    path.write_text(
        json.dumps(
            {
                "kind": "paired_outcome_preference_margin_report_v1",
                "episode_count": pair_id if episode_count is None else episode_count,
                "current_context_episode_count": context_count,
                "reference_context_episode_count": context_count,
                "current_context_coverage": {
                    "episode_count": pair_id if episode_count is None else episode_count,
                    "context_episode_count": context_count,
                    "missing_context_episode_count": max(
                        0,
                        (pair_id if episode_count is None else episode_count) - context_count,
                    ),
                    "empty_opponent_id_episode_count": 0,
                    "missing_context_opponent_policy_ids": missing_context_opponents or [],
                },
                "reference_context_coverage": {
                    "episode_count": pair_id if episode_count is None else episode_count,
                    "context_episode_count": context_count,
                    "missing_context_episode_count": max(
                        0,
                        (pair_id if episode_count is None else episode_count) - context_count,
                    ),
                    "empty_opponent_id_episode_count": 0,
                    "missing_context_opponent_policy_ids": missing_context_opponents or [],
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_preference_mechanistic_gate_passes_when_all_required_groups_improve(tmp_path: Path) -> None:
    pre = _write_report(
        tmp_path / "pre.json",
        margins={"learned_repair": [0.0, 0.0], "b2_preserve": [0.0], "p4_preserve": [0.0]},
    )
    post = _write_report(
        tmp_path / "post.json",
        margins={"learned_repair": [0.2, 0.1], "b2_preserve": [0.05], "p4_preserve": [0.04]},
    )

    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(
            pre_report_json=pre,
            post_report_json=post,
            required_groups=("learned_repair", "b2_preserve", "p4_preserve"),
            require_context=False,
        )
    )

    assert report["passed"] is True
    assert report["summary"]["pair_improved_fraction"] == pytest.approx(1.0)
    assert {group["label"] for group in report["groups"]} == {"learned_repair", "b2_preserve", "p4_preserve"}


def test_preference_mechanistic_gate_fails_when_required_group_drops(tmp_path: Path) -> None:
    pre = _write_report(
        tmp_path / "pre.json",
        margins={"learned_repair": [0.0], "b2_preserve": [0.0], "p4_preserve": [0.0]},
    )
    post = _write_report(
        tmp_path / "post.json",
        margins={"learned_repair": [0.2], "b2_preserve": [-0.01], "p4_preserve": [0.03]},
    )

    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(
            pre_report_json=pre,
            post_report_json=post,
            required_groups=("learned_repair", "b2_preserve", "p4_preserve"),
            require_context=False,
        )
    )

    assert report["passed"] is False
    assert any(str(failure).startswith("min_delta_below") for failure in report["failures"])
    assert any("required_group_mean_delta_below:b2_preserve" in str(failure) for failure in report["failures"])


def test_preference_mechanistic_gate_requires_declared_groups(tmp_path: Path) -> None:
    pre = _write_report(tmp_path / "pre.json", margins={"learned_repair": [0.0]})
    post = _write_report(tmp_path / "post.json", margins={"learned_repair": [0.1]})

    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(
            pre_report_json=pre,
            post_report_json=post,
            required_groups=("learned_repair", "b2_preserve"),
            require_context=False,
        )
    )

    assert report["passed"] is False
    assert "missing_required_groups:b2_preserve" in report["failures"]


def test_preference_mechanistic_gate_can_require_context(tmp_path: Path) -> None:
    pre = _write_report(tmp_path / "pre.json", margins={"learned_repair": [0.0]})
    post = _write_report(tmp_path / "post.json", margins={"learned_repair": [0.1]}, context_count=0)

    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(pre_report_json=pre, post_report_json=post)
    )

    assert report["passed"] is False
    assert "missing_opponent_context" in report["failures"]


def test_preference_mechanistic_gate_requires_full_context_coverage(tmp_path: Path) -> None:
    pre = _write_report(
        tmp_path / "pre.json",
        margins={"learned_repair": [0.0], "b2_preserve": [0.0]},
        context_count=2,
    )
    post = _write_report(
        tmp_path / "post.json",
        margins={"learned_repair": [0.1], "b2_preserve": [0.1]},
        context_count=1,
        episode_count=2,
        missing_context_opponents=["B2 HeuristicPublic"],
    )

    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(
            pre_report_json=pre,
            post_report_json=post,
            required_groups=("learned_repair", "b2_preserve"),
        )
    )

    assert report["passed"] is False
    assert "current_context_episodes_below:1<2" in report["failures"]
    assert "current_missing_context_opponents:B2 HeuristicPublic" in report["failures"]
