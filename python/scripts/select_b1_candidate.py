from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.b1_candidate_selection import (
    DEFAULT_CONFIRM_OPPONENTS,
    DEFAULT_REQUIRED_ANCHORS,
    SELECTED_CANDIDATE_POLICY_ID,
    build_b1_candidate_selection,
    load_reference_anchor_scores,
    publish_b1_baseline_alias,
    publish_selected_candidate_alias,
)


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a B1 checkpoint from saved periodic dev-eval artifacts")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--stack-config", type=Path, default=None)
    parser.add_argument("--required-anchor", action="append", default=None)
    parser.add_argument("--confirm-opponent", action="append", default=None)
    parser.add_argument("--min-required-anchor-score", type=float, default=0.5)
    parser.add_argument("--falloff-warning-threshold", type=float, default=0.05)
    parser.add_argument("--confirm-paired-seeds", type=int, default=64)
    parser.add_argument(
        "--reference-summary-json",
        type=Path,
        default=None,
        help="Optional targeted-confirm summary or score mapping used to report candidate-vs-reference deltas",
    )
    parser.add_argument(
        "--reference-label",
        default="reference",
        help="Human-readable label for --reference-summary-json in the output artifact",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--publish-baseline-alias",
        action="store_true",
        help="Copy the selected snapshot to a pinned b1_noleague_baseline alias in its source run",
    )
    parser.add_argument(
        "--publish-selected-alias",
        action="store_true",
        help="Copy the selected snapshot to a pinned generic selected-candidate alias in its source run",
    )
    parser.add_argument(
        "--selected-alias-policy-id",
        default=SELECTED_CANDIDATE_POLICY_ID,
        help="Policy id to write when --publish-selected-alias is used",
    )
    args = parser.parse_args()

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
            selection_summary={
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
            },
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
            selection_summary={
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
            },
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.output_json)
        return

    print(json.dumps({key: summary[key] for key in ("candidate_count", "warnings")}, indent=2, sort_keys=True))
    if isinstance(selected, dict):
        print(
            "selected "
            f"run={selected['run_name']} snapshot={selected['snapshot_policy_id']} "
            f"update={selected['update_count']} score={selected['selection_score']:.6f} "
            f"required_min={selected['required_anchor_min']:.6f} eligible={selected['eligible']}"
        )
        command = selected.get("confirmation_command")
        if isinstance(command, list):
            print("confirm_command " + _command_text([str(part) for part in command]))
    published = summary.get("published_baseline_alias")
    if isinstance(published, dict):
        print(
            "published_baseline_alias "
            f"policy_id={published['policy_id']} source={published['alias_for_policy_id']} "
            f"update={published['update']}"
        )
    published_selected = summary.get("published_selected_alias")
    if isinstance(published_selected, dict):
        print(
            "published_selected_alias "
            f"policy_id={published_selected['policy_id']} source={published_selected['alias_for_policy_id']} "
            f"update={published_selected['update']}"
        )


if __name__ == "__main__":
    main()
