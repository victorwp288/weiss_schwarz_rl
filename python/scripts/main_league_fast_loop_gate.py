#!/usr/bin/env python3
"""Gate main-league probe escalation before spending game-eval time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.main_league_fast_loop_gate import (
    FAST_LOOP_STAGE_DECISIONS,
    MainLeagueFastLoopGateConfig,
    evaluate_main_league_fast_loop_gate,
    write_main_league_fast_loop_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(FAST_LOOP_STAGE_DECISIONS))
    parser.add_argument("--mechanistic-gate-json", required=True, type=Path)
    parser.add_argument("--target-gate-json", default=None, type=Path)
    parser.add_argument("--drift-gate-json", default=None, type=Path)
    parser.add_argument("--live-progress-gate-json", default=None, type=Path)
    parser.add_argument("--frontier-scorecard-json", default=None, type=Path)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage=str(args.stage),
            mechanistic_gate_json=args.mechanistic_gate_json.resolve(),
            target_gate_json=None if args.target_gate_json is None else args.target_gate_json.resolve(),
            drift_gate_json=None if args.drift_gate_json is None else args.drift_gate_json.resolve(),
            live_progress_gate_json=None
            if args.live_progress_gate_json is None
            else args.live_progress_gate_json.resolve(),
            frontier_scorecard_json=None
            if args.frontier_scorecard_json is None
            else args.frontier_scorecard_json.resolve(),
            candidate_label=args.candidate_label,
        )
    )
    write_main_league_fast_loop_gate(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "stage": report["stage"],
                "passed": report["passed"],
                "failures": report["failures"],
                "required_decision": report["required_decision"],
            },
            sort_keys=True,
        )
    )
    if not bool(report["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
