#!/usr/bin/env python3
"""Aggregate main-league frontier diagnostics into one audit report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.main_league_frontier_audit import (
    MainLeagueFrontierAuditConfig,
    build_main_league_frontier_audit,
    write_main_league_frontier_audit,
    write_main_league_frontier_audit_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--date-token", default="20260521")
    parser.add_argument("--selected-run", default="runs/main_champion_hardneg_interp_u10_repair_a015_20260517")
    parser.add_argument("--selected-policy-id", default="main_interp_repair_a015")
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(
            diagnostics_dir=args.diagnostics_dir,
            output_json=args.output_json,
            date_token=str(args.date_token),
            selected_run=str(args.selected_run),
            selected_policy_id=str(args.selected_policy_id),
            max_entries=int(args.max_entries),
        )
    )
    write_main_league_frontier_audit(args.output_json, report)
    if args.output_md is not None:
        write_main_league_frontier_audit_markdown(args.output_md, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "output_md": None if args.output_md is None else args.output_md.as_posix(),
                "candidate_count": report["candidate_count"],
                "scorecard_entry_count": report["scorecard_entry_count"],
                "gate_entry_count": report["gate_entry_count"],
                "publishable_successor_exists": report["decision"]["publishable_successor_exists"],
                "selected_remains_locked": report["decision"]["selected_remains_locked"],
                "candidates_by_next_stage": report["counts"]["candidates_by_next_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
