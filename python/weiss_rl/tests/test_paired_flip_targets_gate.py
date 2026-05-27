from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.paired_flip_targets_gate import (
    build_paired_flip_targets_gate_config,
    evaluate_paired_flip_targets_gate,
)


def test_paired_flip_targets_gate_passes_sufficient_nonexcluded_coverage(tmp_path: Path) -> None:
    targets = _write_targets(
        tmp_path / "targets.json",
        [
            {"opponent_policy_id": "policy_a", "pair_index": 68},
            {"opponent_policy_id": "policy_b", "pair_index": 70},
        ],
    )

    report = evaluate_paired_flip_targets_gate(
        build_paired_flip_targets_gate_config(
            target_jsons=[targets],
            min_total_targets=2,
            min_target_opponents=2,
            min_distinct_pair_indices=2,
            excluded_pair_indices=[205],
            required_opponents=["policy_a"],
        )
    )

    assert report["passed"] is True
    assert report["summary"]["target_count"] == 2
    assert report["summary"]["target_opponents"] == ["policy_a", "policy_b"]


def test_paired_flip_targets_gate_fails_singleton_and_excluded_pair(tmp_path: Path) -> None:
    targets = _write_targets(
        tmp_path / "targets.json",
        [{"opponent_policy_id": "policy_a", "pair_index": 205}],
    )

    report = evaluate_paired_flip_targets_gate(
        build_paired_flip_targets_gate_config(
            target_jsons=[targets],
            min_total_targets=2,
            min_target_opponents=2,
            min_distinct_pair_indices=2,
            excluded_pair_indices=[205],
            required_opponents=["policy_b"],
        )
    )

    assert report["passed"] is False
    assert "target_count_below:1<2" in report["failures"]
    assert "target_opponent_count_below:1<2" in report["failures"]
    assert "distinct_pair_index_count_below:1<2" in report["failures"]
    assert "excluded_pair_indices_present:205" in report["failures"]
    assert "required_opponents_missing:policy_b" in report["failures"]


def _write_targets(path: Path, targets: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "paired_flip_targets_v1",
                "target_count": len(targets),
                "targets": targets,
            }
        ),
        encoding="utf-8",
    )
    return path
