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

from weiss_rl.experiments import main_league_frontier_scorecard as _frontier_scorecard

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

MAIN_LEAGUE_SENTINEL_OPPONENTS = list(_frontier_scorecard.MAIN_LEAGUE_SENTINEL_OPPONENTS)
MAIN_LEAGUE_FULL13_OPPONENTS = list(_frontier_scorecard.MAIN_LEAGUE_FULL13_OPPONENTS)

OPPONENT_SETS = {
    "default": DEFAULT_OPPONENTS,
    "main_league_full13": MAIN_LEAGUE_FULL13_OPPONENTS,
    "main_league_sentinel": MAIN_LEAGUE_SENTINEL_OPPONENTS,
}

FAST_LOOP_EXACT_PAIRED_SEEDS = {
    "full_confirm64": 64,
    "confirm128": 128,
    "confirm256": 256,
    "publish": 256,
}


def _targeted_eval_job(
    *,
    args: argparse.Namespace,
    paired_seeds: list[int],
    opponent_index: int,
    opponent: str,
    output_dir: Path,
) -> dict:
    job = {
        "stack_config": args.stack_config.as_posix(),
        "run_dir": args.run_dir.as_posix(),
        "snapshot_registry_json": args.snapshot_registry_json.as_posix(),
        "b1_baseline_run_dir": args.b1_baseline_run_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "paired_seeds": paired_seeds,
        "stage1_paired_seeds": int(args.paired_seeds),
        "max_paired_seeds": int(args.paired_seeds),
        "bootstrap_samples": int(args.bootstrap_samples),
        "scheme": "S0",
        "focal_index": 0,
        "opponent_index": opponent_index,
        "focal_policy_id": args.focal_policy_id,
        "opponent_policy_id": opponent,
    }
    god_search = _god_search_payload_from_args(args)
    if god_search is not None:
        job["god_search"] = god_search
    return job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused targeted confirmation eval")
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-registry-json", required=True, type=Path)
    parser.add_argument("--b1-baseline-run-dir", required=True, type=Path)
    parser.add_argument("--focal-policy-id", default="policy_000021")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument(
        "--opponent-set",
        choices=sorted(OPPONENT_SETS),
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
    return parser.parse_args()


def _resolve_paired_seed_file(args: argparse.Namespace, stack) -> tuple[Path, str]:
    explicit_path = getattr(args, "paired_seed_file", None)
    if explicit_path is not None:
        return Path(explicit_path).resolve(), "explicit"
    seed_set = str(getattr(args, "seed_set", "report_eval"))
    if seed_set not in stack.seed_sets:
        raise KeyError(f"seed set not found in stack config: {seed_set}")
    return Path(stack.seed_sets[seed_set]), seed_set


def _resolve_opponents(args: argparse.Namespace) -> list[str]:
    explicit = [item.strip() for item in args.opponent if item.strip()]
    if explicit:
        return explicit
    opponent_set = str(getattr(args, "opponent_set", "default"))
    try:
        return list(OPPONENT_SETS[opponent_set])
    except KeyError as exc:
        raise KeyError(f"unknown opponent set: {opponent_set}") from exc


def _god_search_payload_from_args(args: argparse.Namespace) -> dict | None:
    mode = str(getattr(args, "god_search_mode", "disabled") or "disabled").strip()
    if mode == "disabled":
        return None
    return {
        "mode": mode,
        "top_k": int(getattr(args, "god_search_top_k", 4)),
        "rollouts_per_action": int(getattr(args, "god_search_rollouts_per_action", 1)),
        "max_rollout_decisions": int(getattr(args, "god_search_max_rollout_decisions", 0)),
        "max_search_decisions_per_game": int(getattr(args, "god_search_max_search_decisions_per_game", 0)),
        "rollout_policy": str(getattr(args, "god_search_rollout_policy", "eval") or "eval"),
        "apply_to_focal_only": True,
        "verify_prefix_replay": not bool(getattr(args, "god_search_no_prefix_verify", False)),
        "fail_on_prefix_mismatch": not bool(getattr(args, "god_search_soft_prefix_fail", False)),
        "trace_limit": int(getattr(args, "god_search_trace_limit", 24)),
    }


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
    if int(args.workers) > 1 and not bool(args.allow_parallel_workers):
        raise SystemExit(
            "targeted confirmation eval is deterministic only with --workers 1; "
            "pass --allow-parallel-workers for exploratory non-selection runs"
        )
    sys.path.insert(0, str((Path.cwd() / "python" / "scripts").resolve()))
    from weiss_rl.artifacts.reproducibility import hash_seed_file, parse_seed_file, require_fixed_python_hash_seed
    from weiss_rl.config import load_stack_config

    try:
        require_fixed_python_hash_seed("targeted confirmation eval")
    except RuntimeError as err:
        raise SystemExit(str(err)) from err
    opponents = _resolve_opponents(args)
    _require_fast_loop_gate(args, opponents=opponents)
    stack = load_stack_config(args.stack_config)
    seed_file_path, seed_source = _resolve_paired_seed_file(args, stack)
    paired_seeds = parse_seed_file(seed_file_path)[: int(args.paired_seeds)]
    if len(paired_seeds) < int(args.paired_seeds):
        raise RuntimeError(f"requested {args.paired_seeds} paired seeds, found {len(paired_seeds)} in {seed_file_path}")

    out_dir = args.run_dir / "eval" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = stack.config.evaluation
    print(
        f"targeted confirm start: focal={args.focal_policy_id} rows={len(opponents)} "
        f"paired_seeds={len(paired_seeds)} workers={args.workers}",
        flush=True,
    )

    jobs = []
    for idx, opponent in enumerate(opponents, start=1):
        jobs.append(
            _targeted_eval_job(
                args=args,
                paired_seeds=paired_seeds,
                opponent_index=idx,
                opponent=opponent,
                output_dir=out_dir,
            )
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
        "output_dir": out_dir.as_posix(),
        "paired_seeds": int(args.paired_seeds),
        "seed_file": {
            "path": seed_file_path.as_posix(),
            "sha256": hash_seed_file(seed_file_path),
            "source": seed_source,
        },
        "games_per_row": int(args.paired_seeds) * 2,
        "stack_config": args.stack_config.as_posix(),
        "eval_sampling_algorithm": None
        if evaluation is None
        else str(getattr(evaluation, "eval_sampling_algorithm", "")),
        "model_sampling_temperature": None
        if evaluation is None
        else float(getattr(evaluation, "model_sampling_temperature", 1.0)),
        "god_search": _god_search_payload_from_args(args),
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


def _require_fast_loop_gate(args: argparse.Namespace, *, opponents: list[str]) -> None:
    stage = getattr(args, "fast_loop_stage", None)
    if not stage:
        return
    _validate_fast_loop_eval_request(stage=str(stage), paired_seeds=int(args.paired_seeds), opponents=opponents)
    from weiss_rl.experiments.main_league_fast_loop_gate import (
        MainLeagueFastLoopGateConfig,
        evaluate_main_league_fast_loop_gate,
    )

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage=str(stage),
            mechanistic_gate_json=getattr(args, "mechanistic_gate_json", None),
            target_gate_json=getattr(args, "target_gate_json", None),
            drift_gate_json=getattr(args, "drift_gate_json", None),
            live_progress_gate_json=getattr(args, "live_progress_gate_json", None),
            frontier_scorecard_json=getattr(args, "frontier_scorecard_json", None),
            candidate_label=getattr(args, "fast_loop_candidate_label", None),
        )
    )
    if not bool(report.get("passed")):
        raise SystemExit(
            "main-league fast-loop gate failed before targeted eval: "
            + json.dumps(report.get("failures", []), sort_keys=True)
        )
    print(
        "main-league fast-loop gate passed: "
        + json.dumps(
            {
                "stage": report.get("stage"),
                "required_decision": report.get("required_decision"),
                "candidate_label": report.get("candidate_label"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _validate_fast_loop_eval_request(*, stage: str, paired_seeds: int, opponents: list[str]) -> None:
    if stage == "sentinel":
        _require_exact_opponent_panel(
            stage=stage,
            actual=opponents,
            expected=MAIN_LEAGUE_SENTINEL_OPPONENTS,
            opponent_set_name="main_league_sentinel",
        )
        return

    exact_paired_seeds = FAST_LOOP_EXACT_PAIRED_SEEDS.get(stage)
    if exact_paired_seeds is not None and paired_seeds != exact_paired_seeds:
        raise SystemExit(
            f"fast-loop stage {stage} must run exactly {exact_paired_seeds} paired seeds; "
            f"got --paired-seeds {paired_seeds}"
        )
    if stage in {"full_confirm64", "confirm128", "confirm256", "publish"}:
        _require_exact_opponent_panel(
            stage=stage,
            actual=opponents,
            expected=MAIN_LEAGUE_FULL13_OPPONENTS,
            opponent_set_name="main_league_full13",
        )


def _require_exact_opponent_panel(
    *,
    stage: str,
    actual: list[str],
    expected: list[str],
    opponent_set_name: str,
) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple == expected_tuple:
        return
    missing = [opponent for opponent in expected_tuple if opponent not in actual_tuple]
    extra = [opponent for opponent in actual_tuple if opponent not in expected_tuple]
    raise SystemExit(
        f"fast-loop stage {stage} must use --opponent-set {opponent_set_name} or the exact same opponent panel; "
        f"missing={missing}; extra={extra}"
    )


if __name__ == "__main__":
    main()
