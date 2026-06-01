"""Runtime orchestration for B1 candidate selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import (
    DEFAULT_CONFIRM_OPPONENTS,
    DEFAULT_REQUIRED_ANCHORS,
    build_b1_candidate_selection,
    load_reference_anchor_scores,
    publish_b1_baseline_alias,
    publish_selected_candidate_alias,
)


@dataclass(frozen=True, slots=True)
class SelectB1CandidateResult:
    summary: dict[str, Any]
    output_json: Path | None


def run_select_b1_candidate(args: argparse.Namespace) -> SelectB1CandidateResult:
    reference_scores = (
        None
        if args.reference_summary_json is None
        else load_reference_anchor_scores(args.reference_summary_json.resolve())
    )
    summary = build_b1_candidate_selection(
        args.run_dir,
        stack_config=None if args.stack_config is None else args.stack_config.resolve(),
        required_anchors=tuple(args.required_anchor or DEFAULT_REQUIRED_ANCHORS),
        confirm_opponents=tuple(args.confirm_opponent or DEFAULT_CONFIRM_OPPONENTS),
        min_required_anchor_score=float(args.min_required_anchor_score),
        falloff_warning_threshold=float(args.falloff_warning_threshold),
        confirm_paired_seeds=int(args.confirm_paired_seeds),
        reference_anchor_scores=reference_scores,
        reference_label=str(args.reference_label),
    )
    selected = summary.get("selected")
    if args.publish_baseline_alias:
        if not isinstance(selected, dict):
            raise SystemExit("cannot publish B1 baseline alias: no selected candidate")
        if not bool(selected.get("eligible")):
            raise SystemExit("cannot publish B1 baseline alias: selected candidate is not eligible")
        summary["published_baseline_alias"] = publish_b1_baseline_alias(
            run_dir=Path(str(selected["run_dir"])),
            source_policy_id=str(selected["snapshot_policy_id"]),
            selection_summary=_baseline_alias_selection_summary(summary=summary, selected=selected),
        )
    if args.publish_selected_alias:
        if not isinstance(selected, dict):
            raise SystemExit("cannot publish selected candidate alias: no selected candidate")
        if not bool(selected.get("eligible")):
            raise SystemExit("cannot publish selected candidate alias: selected candidate is not eligible")
        summary["published_selected_alias"] = publish_selected_candidate_alias(
            run_dir=Path(str(selected["run_dir"])),
            source_policy_id=str(selected["snapshot_policy_id"]),
            alias_policy_id=str(args.selected_alias_policy_id),
            selection_summary=_selected_alias_selection_summary(summary=summary, selected=selected),
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SelectB1CandidateResult(summary=summary, output_json=args.output_json)


def baseline_alias_selection_summary(*, summary: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return _baseline_alias_selection_summary(summary=summary, selected=selected)


def selected_alias_selection_summary(*, summary: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return _selected_alias_selection_summary(summary=summary, selected=selected)


def _baseline_alias_selection_summary(*, summary: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_anchors": summary["required_anchors"],
        "min_required_anchor_score": summary["min_required_anchor_score"],
        "selected": {
            key: selected.get(key)
            for key in (
                "run_name",
                "train_policy_id",
                "snapshot_policy_id",
                "update_count",
                "aggregate_score",
                "required_anchor_mean",
                "required_anchor_min",
                "selection_score",
                "confirmatory_dev_eval",
            )
        },
    }


def _selected_alias_selection_summary(*, summary: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_anchors": summary["required_anchors"],
        "min_required_anchor_score": summary["min_required_anchor_score"],
        "selected": {
            key: selected.get(key)
            for key in (
                "run_name",
                "train_policy_id",
                "snapshot_policy_id",
                "update_count",
                "aggregate_score",
                "required_anchor_mean",
                "required_anchor_min",
                "selection_score",
                "selection_score_source",
                "selection_confirmation_summary_path",
            )
        },
    }
