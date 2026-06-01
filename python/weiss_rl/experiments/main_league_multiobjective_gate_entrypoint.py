#!/usr/bin/env python3
"""Evaluate the thesis main-league candidate against fixed and learned-opponent gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.main_league_multiobjective_gate import (
    FIXED_THESIS_OPPONENTS,
    MultiObjectiveGateConfig,
    evaluate_main_league_multiobjective_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main-league multi-objective gate")
    parser.add_argument("--candidate-summary-json", action="append", required=True, type=Path)
    parser.add_argument("--reference-summary-json", action="append", default=[], type=Path)
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument("--learned-opponent", action="append", default=[])
    parser.add_argument(
        "--opponent-alias",
        action="append",
        default=[],
        help="Map a reference opponent id to a candidate opponent id as OLD=NEW.",
    )
    parser.add_argument("--min-fixed-score", type=float, default=0.5)
    parser.add_argument("--max-fixed-reference-drop", type=float, default=0.0)
    parser.add_argument("--min-learned-score", type=float, default=0.5)
    parser.add_argument("--min-learned-mean", type=float, default=0.5)
    parser.add_argument("--min-learned-reference-delta", type=float, default=0.0)
    parser.add_argument("--max-learned-reference-drop", type=float, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aliases = _parse_aliases(args.opponent_alias)
    summary = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=tuple(path.resolve() for path in args.candidate_summary_json),
            reference_summary_jsons=tuple(path.resolve() for path in args.reference_summary_json),
            fixed_opponents=tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS),
            learned_opponents=tuple(str(item) for item in args.learned_opponent),
            opponent_aliases=aliases,
            min_fixed_score=float(args.min_fixed_score),
            max_fixed_reference_drop=float(args.max_fixed_reference_drop),
            min_learned_score=float(args.min_learned_score),
            min_learned_mean=float(args.min_learned_mean),
            min_learned_reference_delta=float(args.min_learned_reference_delta),
            max_learned_reference_drop=args.max_learned_reference_drop,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "output_json": args.output_json.as_posix()}, sort_keys=True))
    if not bool(summary["passed"]):
        raise SystemExit(2)


def _parse_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--opponent-alias must be OLD=NEW, got: {value!r}")
        old, new = value.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise SystemExit(f"--opponent-alias must be OLD=NEW, got: {value!r}")
        aliases[old] = new
    return aliases


if __name__ == "__main__":
    main()
