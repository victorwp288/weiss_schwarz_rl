#!/usr/bin/env python3
"""Run a focused high-seed targeted eval using the repo's parallel final-eval worker.

This script is intended to be copied into the remote WSRL repo and run from
the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


DEFAULT_OPPONENTS = [
    "B0 RandomLegal",
    "B1 NoLeague baseline",
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
    "seed_2aaa98fc69_seed_ae46265326_seed_bafd7a640b_policy_000011",
    "seed_2aaa98fc69_seed_ae46265326_policy_000012",
    "seed_2aaa98fc69_seed_ae46265326_policy_000014",
    "seed_2aaa98fc69_seed_ae46265326_policy_000015",
    "seed_2aaa98fc69_seed_ae46265326_policy_000016",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused targeted confirmation eval")
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-registry-json", required=True, type=Path)
    parser.add_argument("--b1-baseline-run-dir", required=True, type=Path)
    parser.add_argument("--focal-policy-id", default="policy_000021")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--paired-seeds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-subdir", default="targeted_confirm64_p21")
    return parser.parse_args()


def _targeted_worker(job: dict) -> dict:
    from parallel_final_eval import _worker  # type: ignore

    result = _worker(job)
    summary_payload = result["summary"]
    summary = summary_payload["summary"]
    uncertainty = summary_payload["uncertainty"]
    games = int(summary.get("games", 0))
    wins = int(summary.get("wins", 0))
    return {
        "focal_policy_id": result["focal_policy_id"],
        "opponent_policy_id": result["opponent_policy_id"],
        "paired_seeds": int(uncertainty.get("paired_seed_count", len(result.get("used_paired_seeds", ())))),
        "games": games,
        "wins": wins,
        "losses": int(summary.get("losses", 0)),
        "draws": int(summary.get("draws", 0)),
        "mean": float(uncertainty.get("mean", wins / games if games else 0.0)),
        "ci_low": float(uncertainty.get("ci_low", 0.0)),
        "ci_high": float(uncertainty.get("ci_high", 0.0)),
        "prob_gt_half": float(uncertainty.get("prob_gt_half", 0.0)),
        "truncations": int(summary.get("truncations", 0)),
        "engine_errors": int(summary.get("engine_errors", 0)),
        "summary_path": (result["matchup_dir"] / "matchup_summary.json").as_posix(),
        "diagnostics_path": (result["matchup_dir"] / "diagnostics.json").as_posix(),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str((Path.cwd() / "python" / "scripts").resolve()))
    from weiss_rl.config import load_stack_config
    from weiss_rl.repro import parse_seed_file

    opponents = [item.strip() for item in args.opponent if item.strip()] or DEFAULT_OPPONENTS
    stack = load_stack_config(args.stack_config)
    paired_seeds = parse_seed_file(stack.seed_sets["report_eval"])[: int(args.paired_seeds)]
    if len(paired_seeds) < int(args.paired_seeds):
        raise RuntimeError(f"requested {args.paired_seeds} paired seeds, found {len(paired_seeds)}")

    out_dir = args.run_dir / "eval" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"targeted confirm start: focal={args.focal_policy_id} rows={len(opponents)} "
        f"paired_seeds={len(paired_seeds)} workers={args.workers}",
        flush=True,
    )

    jobs = []
    for idx, opponent in enumerate(opponents, start=1):
        jobs.append(
            {
                "stack_config": args.stack_config.as_posix(),
                "run_dir": args.run_dir.as_posix(),
                "snapshot_registry_json": args.snapshot_registry_json.as_posix(),
                "b1_baseline_run_dir": args.b1_baseline_run_dir.as_posix(),
                "paired_seeds": paired_seeds,
                "stage1_paired_seeds": int(args.paired_seeds),
                "max_paired_seeds": int(args.paired_seeds),
                "bootstrap_samples": int(args.bootstrap_samples),
                "scheme": "S0",
                "focal_index": 0,
                "opponent_index": idx,
                "focal_policy_id": args.focal_policy_id,
                "opponent_policy_id": opponent,
            }
        )

    results_by_opp = {}
    started = time.time()
    if int(args.workers) <= 1:
        for job in jobs:
            try:
                result = _targeted_worker(job)
            except Exception:
                traceback.print_exc()
                raise
            opponent = job["opponent_policy_id"]
            results_by_opp[opponent] = result
            print(
                f"done {len(results_by_opp)}/{len(jobs)} {args.focal_policy_id} vs {opponent} "
                f"mean={result.get('mean')} wins={result.get('wins')}/{result.get('games')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(_targeted_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception:
                    traceback.print_exc()
                    raise
                opponent = job["opponent_policy_id"]
                results_by_opp[opponent] = result
                print(
                    f"done {len(results_by_opp)}/{len(jobs)} {args.focal_policy_id} vs {opponent} "
                    f"mean={result.get('mean')} wins={result.get('wins')}/{result.get('games')}",
                    flush=True,
                )

    rows = []
    for job in jobs:
        opponent = job["opponent_policy_id"]
        result = results_by_opp[opponent]
        rows.append(
            {
                "focal_policy_id": args.focal_policy_id,
                "opponent_policy_id": opponent,
                "paired_seeds": result.get("paired_seeds"),
                "games": result.get("games"),
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "draws": result.get("draws"),
                "mean": result.get("mean"),
                "ci_low": result.get("ci_low"),
                "ci_high": result.get("ci_high"),
                "prob_gt_half": result.get("prob_gt_half"),
                "truncations": result.get("truncations"),
                "engine_errors": result.get("engine_errors"),
                "summary_path": result.get("summary_path"),
                "diagnostics_path": result.get("diagnostics_path"),
            }
        )

    anchor_rows = rows[:5]
    league_rows = rows[5:]
    summary = {
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "focal_policy_id": args.focal_policy_id,
        "paired_seeds": int(args.paired_seeds),
        "games_per_row": int(args.paired_seeds) * 2,
        "rows": rows,
        "overall": {"wins": sum(row["wins"] for row in rows), "games": sum(row["games"] for row in rows)},
        "anchor_subset": {
            "wins": sum(row["wins"] for row in anchor_rows),
            "games": sum(row["games"] for row in anchor_rows),
        },
        "legacy_subset": {
            "wins": sum(row["wins"] for row in league_rows),
            "games": sum(row["games"] for row in league_rows),
        },
    }
    for key in ("overall", "anchor_subset", "legacy_subset"):
        summary[key]["mean"] = summary[key]["wins"] / summary[key]["games"] if summary[key]["games"] else None
    summary_path = out_dir / f"targeted_confirm{args.paired_seeds}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary {summary_path}", flush=True)
    print(json.dumps(summary["overall"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
