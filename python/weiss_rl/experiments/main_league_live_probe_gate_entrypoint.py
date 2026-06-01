#!/usr/bin/env python3
"""Gate live main-league probes before sentinel game eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.main_league_live_probe_gate import (
    MainLeagueLiveProbeGateConfig,
    evaluate_main_league_live_probe_gate,
    write_main_league_live_probe_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-progress-summary-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-champion-envs", type=float, default=1.0)
    parser.add_argument("--min-hard-negative-envs", type=float, default=1.0)
    parser.add_argument("--min-heuristic-public-envs", type=float, default=1.0)
    parser.add_argument("--min-heuristic-public-variant-envs", type=float, default=1.0)
    parser.add_argument("--min-noleague-baseline-envs", type=float, default=1.0)
    parser.add_argument("--min-champion-pool-size", type=float, default=1.0)
    parser.add_argument("--min-hard-negative-pool-size", type=float, default=1.0)
    parser.add_argument("--required-sampled-policy", action="append", default=[])
    parser.add_argument("--min-required-sampled-policy-envs", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_main_league_live_probe_gate(
        MainLeagueLiveProbeGateConfig(
            league_progress_summary_json=args.league_progress_summary_json,
            min_champion_envs=float(args.min_champion_envs),
            min_hard_negative_envs=float(args.min_hard_negative_envs),
            min_heuristic_public_envs=float(args.min_heuristic_public_envs),
            min_heuristic_public_variant_envs=float(args.min_heuristic_public_variant_envs),
            min_noleague_baseline_envs=float(args.min_noleague_baseline_envs),
            min_champion_pool_size=float(args.min_champion_pool_size),
            min_hard_negative_pool_size=float(args.min_hard_negative_pool_size),
            required_sampled_policies=tuple(str(item) for item in args.required_sampled_policy),
            min_required_sampled_policy_envs=float(args.min_required_sampled_policy_envs),
        )
    )
    write_main_league_live_probe_gate(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": bool(report["passed"]),
                "failures": report["failures"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
