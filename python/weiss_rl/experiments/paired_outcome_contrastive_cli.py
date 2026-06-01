"""CLI parser for paired-outcome contrastive dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_paired_outcome_contrastive_parser() -> argparse.ArgumentParser:
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


__all__ = ["build_paired_outcome_contrastive_parser"]
