"""Parser construction for B1 candidate selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.experiments.b1_candidate_selection import SELECTED_CANDIDATE_POLICY_ID


def build_select_b1_candidate_parser() -> argparse.ArgumentParser:
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
    return parser
