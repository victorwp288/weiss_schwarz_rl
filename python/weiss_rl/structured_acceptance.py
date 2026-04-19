"""Structured-v2 baseline and acceptance helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_B2_ANCHOR = "B2 HeuristicPublic"


@dataclass(frozen=True, slots=True)
class StructuredBaselineContract:
    baseline_run_dir: str
    baseline_update: int
    audit_summary_path: str
    aggregate_score: float
    anchor_scores: dict[str, float]
    mismatch_baseline: dict[str, int]
    dominant_exact_pair: dict[str, Any]
    acceptance_targets: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_dir": self.baseline_run_dir,
            "baseline_update": self.baseline_update,
            "audit_summary_path": self.audit_summary_path,
            "aggregate_score": self.aggregate_score,
            "anchor_scores": dict(self.anchor_scores),
            "mismatch_baseline": dict(self.mismatch_baseline),
            "dominant_exact_pair": dict(self.dominant_exact_pair),
            "acceptance_targets": dict(self.acceptance_targets),
        }


def build_structured_baseline_contract(
    *,
    baseline_run_dir: Path,
    baseline_update: int,
    dev_eval_summary: dict[str, Any],
    audit_summary_path: Path,
    audit_summary: dict[str, Any],
    b2_anchor_name: str = _DEFAULT_B2_ANCHOR,
) -> StructuredBaselineContract:
    aggregate_score = float(dev_eval_summary.get("aggregate_score", 0.0))
    anchor_scores = {str(name): float(score) for name, score in dict(dev_eval_summary.get("anchor_scores", {})).items()}
    if b2_anchor_name not in anchor_scores:
        raise ValueError(f"dev_eval summary is missing anchor score for {b2_anchor_name!r}")

    top_family_pairs = list(audit_summary.get("top_family_pairs", ()))
    top_action_label_pairs = list(audit_summary.get("top_action_label_pairs", ()))
    main_move_to_pass = _find_counter(
        top_family_pairs,
        policy_a_family="main_move",
        policy_b_family="pass",
    )
    main_move_to_play = _find_counter(
        top_family_pairs,
        policy_a_family="main_move",
        policy_b_family="main_play_character",
    )
    exact_main_move_02_to_pass = _find_counter(
        top_action_label_pairs,
        policy_a_action_label="main_move(from_slot=0, to_slot=2)",
        policy_b_action_label="pass",
    )
    dominant_exact_pair = dict(top_action_label_pairs[0]) if top_action_label_pairs else {}

    acceptance_targets = {
        "u120": {
            "no_seed_zero_vs_b2": True,
            "max_main_move_to_pass": _reduced_target(main_move_to_pass, reduction=0.60),
            "max_main_move_to_main_play_character": _reduced_target(
                main_move_to_play,
                reduction=0.60,
            ),
            "forbid_top_exact_pair": {
                "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                "policy_b_action_label": "pass",
            },
        },
        "u300": {
            "min_aggregate_score": aggregate_score + 0.10,
            "min_anchor_scores": {
                "B2 HeuristicPublic": 0.25,
            },
            "min_mean_anchor_score_delta": 0.10,
            "max_anchor_regressions": {
                name: max(score - 0.05, 0.0) for name, score in anchor_scores.items() if name != b2_anchor_name
            },
            "min_b2_score_per_seed": 0.10,
        },
    }

    return StructuredBaselineContract(
        baseline_run_dir=baseline_run_dir.resolve().as_posix(),
        baseline_update=int(baseline_update),
        audit_summary_path=audit_summary_path.resolve().as_posix(),
        aggregate_score=aggregate_score,
        anchor_scores=anchor_scores,
        mismatch_baseline={
            "main_move_to_pass": main_move_to_pass,
            "main_move_to_main_play_character": main_move_to_play,
            "exact_main_move_0_2_to_pass": exact_main_move_02_to_pass,
        },
        dominant_exact_pair=dominant_exact_pair,
        acceptance_targets=acceptance_targets,
    )


def _find_counter(items: list[dict[str, Any]], **match: str) -> int:
    for item in items:
        if all(str(item.get(key, "")) == value for key, value in match.items()):
            return int(item.get("count", 0))
    return 0


def _reduced_target(baseline_count: int, *, reduction: float) -> int:
    if baseline_count <= 0:
        return 0
    keep_fraction = max(0.0, 1.0 - float(reduction))
    return max(0, math.floor(float(baseline_count) * keep_fraction))


__all__ = ["StructuredBaselineContract", "build_structured_baseline_contract"]
