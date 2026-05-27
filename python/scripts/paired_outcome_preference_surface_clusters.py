#!/usr/bin/env python3
"""Classify paired-outcome preference rows by public-surface separability."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_surface_clusters import (
    PairedOutcomePreferenceSurfaceClusterConfig,
    build_paired_outcome_preference_surface_cluster_report,
    write_paired_outcome_preference_surface_cluster_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--spec-bundle-json",
        type=Path,
        default=None,
        help="Optional spec_bundle.json used to decode action ids into families and slots.",
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        help="Optional training config used to read model.opponent_context_policy_ids.",
    )
    parser.add_argument(
        "--opponent-context-policy-id",
        action="append",
        default=[],
        help="Additional policy id that maps to a nonzero opponent-context index.",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_surface_cluster_report(
        PairedOutcomePreferenceSurfaceClusterConfig(
            dataset_path=args.dataset,
            spec_bundle_json=args.spec_bundle_json,
            stack_config_path=args.stack_config,
            opponent_context_policy_ids=tuple(args.opponent_context_policy_id),
            max_examples=int(args.max_examples),
        )
    )
    write_paired_outcome_preference_surface_cluster_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "aligned_different_action_count": report["aligned_different_action_count"],
                "same_public_surface_different_action_count": report[
                    "same_public_surface_different_action_count"
                ],
                "surface_conflict_count": report["surface_conflict_count"],
                "opponent_context_resolvable_conflict_count": report[
                    "opponent_context_resolvable_conflict_count"
                ],
                "opponent_context_required_missing_mapping_count": report[
                    "opponent_context_required_missing_mapping_count"
                ],
                "replay_only_required_conflict_count": report["replay_only_required_conflict_count"],
                "public_surface_separable": report["public_surface_separable"],
                "unconditioned_replay_safe": report["unconditioned_replay_safe"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
