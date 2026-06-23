from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.eval.targeted_confirm import opponents as _opponents


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused targeted confirmation eval")
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-registry-json", required=True, type=Path)
    parser.add_argument("--b1-baseline-run-dir", required=True, type=Path)
    parser.add_argument("--focal-policy-id", default="policy_000021")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument(
        "--opponent-set",
        choices=sorted(_opponents.OPPONENT_SETS),
        default="default",
        help=(
            "Named opponent set used when --opponent is omitted. "
            "Use main_league_sentinel for cheap B2/B4 plus learned-row triage before full confirm. "
            "Use main_league_full13 for the current B0-B4 plus b8 champion/hard-negative panel."
        ),
    )
    parser.add_argument("--paired-seeds", type=int, default=64)
    parser.add_argument(
        "--seed-set",
        default="report_eval",
        help="Stack seed-set name to use for paired seeds when --paired-seed-file is not provided.",
    )
    parser.add_argument(
        "--paired-seed-file",
        type=Path,
        default=None,
        help="Explicit paired seed file for diagnostic non-report surfaces.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--allow-parallel-workers",
        action="store_true",
        help="Allow workers >1. Parallel simulator eval is experimental and should not be used for checkpoint selection.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-subdir", default="targeted_confirm64_p21")
    parser.add_argument(
        "--fast-loop-stage",
        choices=("sentinel", "full_confirm64", "confirm128", "confirm256", "publish"),
        default=None,
        help="Require the thesis main-league fast-loop gate before starting this eval stage.",
    )
    parser.add_argument(
        "--mechanistic-gate-json",
        default=None,
        type=Path,
        help="Mechanistic gate JSON required by --fast-loop-stage.",
    )
    parser.add_argument(
        "--drift-gate-json",
        default=None,
        type=Path,
        help="Optional trajectory drift gate JSON that must pass before --fast-loop-stage eval starts.",
    )
    parser.add_argument(
        "--live-progress-gate-json",
        default=None,
        type=Path,
        help="Optional live league exposure gate JSON that can satisfy the sentinel pre-eval diagnostic gate.",
    )
    parser.add_argument(
        "--target-gate-json",
        default=None,
        type=Path,
        help="Optional paired-flip target coverage gate JSON that must pass before --fast-loop-stage eval starts.",
    )
    parser.add_argument(
        "--frontier-scorecard-json",
        default=None,
        type=Path,
        help="Frontier scorecard JSON required for full_confirm64/confirm128/confirm256 escalation.",
    )
    parser.add_argument(
        "--fast-loop-candidate-label",
        default=None,
        help="Candidate label to select when the frontier scorecard contains multiple entries.",
    )
    parser.add_argument(
        "--god-search-mode",
        choices=("disabled", "same_world_prefix_rollout"),
        default="disabled",
        help=(
            "Enable exploratory decision-time search for the focal policy. "
            "same_world_prefix_rollout replays the current episode prefix and must be labeled as same-world search."
        ),
    )
    parser.add_argument("--god-search-top-k", type=int, default=4)
    parser.add_argument("--god-search-rollouts-per-action", type=int, default=1)
    parser.add_argument(
        "--god-search-max-rollout-decisions",
        type=int,
        default=0,
        help="Per-candidate rollout horizon after the forced root action; 0 rolls to terminal/truncation.",
    )
    parser.add_argument(
        "--god-search-max-search-decisions-per-game",
        type=int,
        default=0,
        help="Maximum focal decisions searched per game; 0 searches every eligible focal decision.",
    )
    parser.add_argument("--god-search-rollout-policy", choices=("eval", "argmax", "sample"), default="eval")
    parser.add_argument("--god-search-no-prefix-verify", action="store_true")
    parser.add_argument("--god-search-soft-prefix-fail", action="store_true")
    parser.add_argument("--god-search-trace-limit", type=int, default=24)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


__all__ = ["build_arg_parser", "parse_args"]
