from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.experiments.toy_public_demo import (
    PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
    PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
)


def add_eval_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stack-config",
        type=Path,
        required=True,
        help="Path to the stack config used for contract checks and evaluation settings",
    )
    parser.add_argument(
        "--spec-hash",
        type=str,
        default="",
        help="Expected compatibility spec hash or full spec bundle SHA-256 for contract validation",
    )
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional startup banner/log label only; not persisted in summary exports",
    )
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)


def add_canonical_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Canonical run directory for simulator-backed evaluation or staged public-demo artifacts",
    )
    parser.add_argument(
        "--final-eval-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for public-demo artifacts, or the canonical "
            "<run-dir>/eval/final_eval path for non-demo runs"
        ),
    )
    parser.add_argument(
        "--policy-id",
        action="append",
        default=None,
        help="Explicit policy ID to evaluate in canonical non-demo mode (repeatable)",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON for deterministic policy-set resolution in canonical non-demo mode",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON for deterministic policy-set resolution in canonical non-demo mode",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing the real B1 no-league baseline artifacts when the selected policy set includes B1"
        ),
    )
    parser.add_argument(
        "--paired-seed-limit",
        type=int,
        default=None,
        help="Optional cap on the number of report_eval paired seeds used in canonical non-demo mode",
    )
    parser.add_argument(
        "--stage1-paired-seeds",
        type=int,
        default=None,
        help="Optional override for stage-1 paired seeds in canonical non-demo mode",
    )
    parser.add_argument(
        "--max-paired-seeds",
        type=int,
        default=None,
        help="Optional override for stage-2 max paired seeds in canonical non-demo mode",
    )
    parser.add_argument(
        "--skip-metagame",
        action="store_true",
        help="Skip metagame sensitivity generation in canonical non-demo mode",
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=None,
        help="Optional study-only metagame/sensitivity config (defaults to configs/study/metagame_sensitivity.yaml)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip paper figure rendering in canonical non-demo mode",
    )
    parser.add_argument(
        "--skip-readiness",
        action="store_true",
        help="Skip paper-readiness auditing in canonical non-demo mode",
    )
    parser.add_argument(
        "--git-commit-override",
        type=str,
        default="",
        help="Optional 40-hex git commit shown in eval logs when manifest provenance is missing; never persisted",
    )


def add_public_demo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Run the built-in public-safe toy final-eval path instead of canonical simulator-backed evaluation.",
    )
    parser.add_argument(
        "--public-demo-paired-seeds",
        type=int,
        default=PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
        help="Paired seed count for public-demo final_eval generation",
    )
    parser.add_argument(
        "--public-demo-bootstrap-samples",
        type=int,
        default=PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count for public-demo final_eval generation",
    )


def add_summary_only_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=None,
        help="Existing seat-swapped episodes.jsonl to summarize (no rollout generation)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Output path for summary JSON export in summary-only mode",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output path for summary CSV export in summary-only mode",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        help="Output path for seat diagnostics JSON export in summary-only mode",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for uncertainty",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed for summary-only mode")


__all__ = [
    "add_canonical_eval_arguments",
    "add_eval_common_arguments",
    "add_public_demo_arguments",
    "add_summary_only_arguments",
]
