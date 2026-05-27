from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_contrastive import (
    PairedOutcomeContrastiveBuildConfig,
    PairedOutcomeInspectionConfig,
    build_paired_outcome_contrastive_dataset,
    inspect_paired_outcome_sources,
    sources_from_paired_flip_summary,
    write_paired_outcome_contrastive_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired-swing contrastive dataset from paired-outcome winner trajectories. "
            "Recorded actions are the positive winner actions; policy B top actions from replay inspections "
            "become teacher_action negatives."
        )
    )
    parser.add_argument("--source-summary-json", type=Path, required=True, help="Paired-flip dataset summary JSON")
    parser.add_argument(
        "--source-role",
        required=True,
        help="Role label for these sources, for example fixed_preserve or learned_repair",
    )
    parser.add_argument(
        "--include-source-label",
        action="append",
        default=[],
        help="Optional source_label or opponent_policy_id filter. Repeat to select multiple sources.",
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Stack config for replay inspection")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir for spec bundle and policy resolution")
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON for resolving policy-a and policy-b during replay inspection.",
    )
    parser.add_argument("--policy-a", required=True, help="Winner policy recorded in the source trajectories")
    parser.add_argument("--policy-b", required=True, help="Losing/reference policy whose top action becomes negative")
    parser.add_argument(
        "--output-run-dir", type=Path, required=True, help="Directory for inspection and source artifacts"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output paired-swing .npz dataset")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary path. Defaults to output path with .summary.json suffix.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100_000,
        help="Replay-inspector top-difference rows to retain per bundle; high values approximate all rows.",
    )
    parser.add_argument("--top-actions", type=int, default=3, help="Top legal actions retained per inspected row")
    parser.add_argument(
        "--min-total-variation",
        type=float,
        default=0.0,
        help="Minimum policy-distribution total variation for a row to become a contrastive pair",
    )
    parser.add_argument("--max-rows-per-bundle", type=int, default=None, help="Optional override-row cap per bundle")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional total override-row cap")
    parser.add_argument("--max-bundles-per-source", type=int, default=None, help="Optional source bundle cap")
    parser.add_argument(
        "--accept-snapshot-config-hash",
        action="append",
        default=[],
        help="Additional config_hash256 accepted by replay policy loading. Repeatable.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute inspections even when the target inspection JSON already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    sources = sources_from_paired_flip_summary(
        args.source_summary_json,
        source_role=args.source_role,
        output_dir=args.output_run_dir / "sources",
        include_source_labels=tuple(args.include_source_label),
    )
    contrastive_sources, inspection_summary = inspect_paired_outcome_sources(
        PairedOutcomeInspectionConfig(
            sources=sources,
            stack_config=args.stack_config,
            run_dir=args.run_dir,
            snapshot_registry_json=args.snapshot_registry_json,
            policy_a=str(args.policy_a),
            policy_b=str(args.policy_b),
            top_k=int(args.top_k),
            top_actions=int(args.top_actions),
            accepted_snapshot_config_hashes=tuple(args.accept_snapshot_config_hash or ()),
            max_bundles_per_source=args.max_bundles_per_source,
            resume=not bool(args.no_resume),
        )
    )
    dataset, dataset_summary = build_paired_outcome_contrastive_dataset(
        PairedOutcomeContrastiveBuildConfig(
            sources=contrastive_sources,
            output_dataset=args.output,
            min_total_variation=float(args.min_total_variation),
            max_rows_per_bundle=args.max_rows_per_bundle,
            max_rows=args.max_rows,
            positive_action_source="actions",
            negative_action_source="teacher_action",
        )
    )
    summary = {
        "kind": "paired_outcome_contrastive_dataset_cli_v1",
        "source_summary_json": args.source_summary_json.as_posix(),
        "source_role": str(args.source_role),
        "stack_config": args.stack_config.as_posix(),
        "run_dir": args.run_dir.as_posix(),
        "snapshot_registry_json": None
        if args.snapshot_registry_json is None
        else args.snapshot_registry_json.as_posix(),
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "output": args.output.as_posix(),
        "inspection_summary": inspection_summary,
        "dataset_summary": dataset_summary,
    }
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    write_paired_outcome_contrastive_summary(summary_path, summary)
    print(
        "Paired-outcome contrastive dataset written to "
        f"{args.output} with {dataset.metadata['train_rows']} train rows, "
        f"{dataset.metadata['bundle_count']} bundles, and "
        f"{dataset.metadata['paired_outcome_contrastive_generation']['distinct_train_rows']} distinct pairs; "
        f"summary written to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
