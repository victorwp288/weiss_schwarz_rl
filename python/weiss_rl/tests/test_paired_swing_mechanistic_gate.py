from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.paired_swing_mechanistic_gate import (
    PairedSwingMechanisticGateConfig,
    evaluate_paired_swing_mechanistic_gate,
)


def test_paired_swing_mechanistic_gate_passes_clean_top_action_improvement(tmp_path: Path) -> None:
    pre = _report(
        [
            _row(label="fixed_preserve_B2", pre_margin=-0.2, top_action=2, positive_rank=2),
            _row(label="learned_repair_p3", pre_margin=-0.1, top_action=2, positive_rank=2),
        ],
        mean=-0.15,
        min_margin=-0.2,
    )
    post = _report(
        [
            _row(label="fixed_preserve_B2", pre_margin=0.1, top_action=1, positive_rank=1),
            _row(label="learned_repair_p3", pre_margin=0.2, top_action=1, positive_rank=1),
        ],
        mean=0.15,
        min_margin=0.1,
    )
    pre_path = _write_json(tmp_path / "pre.json", pre)
    post_path = _write_json(tmp_path / "post.json", post)

    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=pre_path,
            post_report_json=post_path,
            max_positive_rank_worsened_fraction=0.0,
        )
    )

    assert report["passed"] is True
    assert report["summary"]["top_positive_delta"] == 2
    assert report["summary"]["positive_rank_improved"] == 2


def test_paired_swing_mechanistic_gate_fails_when_top_action_gets_worse(tmp_path: Path) -> None:
    pre = _report(
        [
            _row(label="fixed_preserve_B2", pre_margin=0.1, top_action=1, positive_rank=1),
            _row(label="learned_repair_p3", pre_margin=-0.1, top_action=2, positive_rank=2),
        ],
        mean=0.0,
        min_margin=-0.1,
    )
    post = _report(
        [
            _row(label="fixed_preserve_B2", pre_margin=0.2, top_action=2, positive_rank=2),
            _row(label="learned_repair_p3", pre_margin=0.1, top_action=1, positive_rank=1),
        ],
        mean=0.15,
        min_margin=0.1,
    )
    pre_path = _write_json(tmp_path / "pre.json", pre)
    post_path = _write_json(tmp_path / "post.json", post)

    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=pre_path,
            post_report_json=post_path,
            max_positive_rank_worsened_fraction=0.0,
        )
    )

    assert report["passed"] is False
    assert "top_positive_delta_below:0<0" not in report["failures"]
    assert any(item.startswith("positive_rank_worsened_fraction_above") for item in report["failures"])


def test_paired_swing_mechanistic_gate_fails_protected_label_drop(tmp_path: Path) -> None:
    pre = _report([_row(label="rawext256_fixed_preserve_B2", pre_margin=0.1)])
    post = _report([_row(label="rawext256_fixed_preserve_B2", pre_margin=0.05)])
    pre_path = _write_json(tmp_path / "pre.json", pre)
    post_path = _write_json(tmp_path / "post.json", post)

    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=pre_path,
            post_report_json=post_path,
            min_mean_delta=-1.0,
            min_min_delta=-1.0,
            min_row_improved_fraction=0.0,
            max_row_worsened_fraction=1.0,
            max_positive_rank_worsened_fraction=1.0,
        )
    )

    assert report["passed"] is False
    assert report["failures"] == ["protected_label_mean_drop:rawext256_fixed_preserve_B2"]


def test_paired_swing_mechanistic_gate_fails_missing_decision_fields(tmp_path: Path) -> None:
    pre = _report([_row(label="learned_repair_p3", pre_margin=0.1)])
    post = _report([_row(label="learned_repair_p3", pre_margin=0.2)])
    del post["rows"][0]["top_action"]
    pre_path = _write_json(tmp_path / "pre.json", pre)
    post_path = _write_json(tmp_path / "post.json", post)

    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=pre_path,
            post_report_json=post_path,
            min_row_improved_fraction=0.0,
        )
    )

    assert report["passed"] is False
    assert "missing_decision_fields:1" in report["failures"]


def test_paired_swing_mechanistic_gate_requires_full_context_coverage(tmp_path: Path) -> None:
    pre = _report([_row(label="fixed_preserve_B2", pre_margin=0.0)], context_count=1, episode_count=1)
    post = _report(
        [_row(label="fixed_preserve_B2", pre_margin=0.2)],
        context_count=0,
        episode_count=1,
        missing_context_opponents=["B2 HeuristicPublic"],
    )
    pre_path = _write_json(tmp_path / "pre.json", pre)
    post_path = _write_json(tmp_path / "post.json", post)

    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=pre_path,
            post_report_json=post_path,
            min_row_improved_fraction=0.0,
        )
    )

    assert report["passed"] is False
    assert "missing_opponent_context" in report["failures"]
    assert "context_episodes_below:0<1" in report["failures"]
    assert "missing_context_opponents:B2 HeuristicPublic" in report["failures"]


def _report(
    rows: list[dict[str, object]],
    *,
    mean: float | None = None,
    min_margin: float | None = None,
    context_count: int | None = None,
    episode_count: int | None = None,
    missing_context_opponents: list[str] | None = None,
) -> dict:
    margins = [float(row["positive_minus_negative_logp"]) for row in rows]
    resolved_episode_count = len(rows) if episode_count is None else episode_count
    resolved_context_count = len(rows) if context_count is None else context_count
    return {
        "kind": "paired_swing_context_margin_report_v1",
        "episode_count": resolved_episode_count,
        "context_episode_count": resolved_context_count,
        "context_coverage": {
            "episode_count": resolved_episode_count,
            "context_episode_count": resolved_context_count,
            "missing_context_episode_count": max(0, resolved_episode_count - resolved_context_count),
            "empty_opponent_id_episode_count": 0,
            "missing_context_opponent_policy_ids": missing_context_opponents or [],
        },
        "row_count": len(rows),
        "positive_margin_mean": sum(margins) / max(len(margins), 1) if mean is None else mean,
        "positive_margin_min": min(margins) if min_margin is None and margins else min_margin,
        "rows": rows,
    }


def _row(
    *,
    label: str,
    pre_margin: float,
    top_action: int = 1,
    positive_rank: int = 1,
) -> dict[str, object]:
    return {
        "step_index": 0,
        "episode_index": len(label),
        "source_dataset_label": label,
        "source_opponent_policy_id": label,
        "positive_action": 1,
        "negative_action": 2,
        "positive_minus_negative_logp": pre_margin,
        "top_action": top_action,
        "positive_rank": positive_rank,
        "negative_rank": 2 if positive_rank == 1 else 1,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
