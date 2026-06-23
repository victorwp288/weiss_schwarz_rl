#!/usr/bin/env python3
"""Run a focused high-seed targeted eval using the repo's parallel final-eval worker.

This script is intended to be copied into the remote WSRL repo and run from
the repository root.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from weiss_rl.eval.final.worker_runtime import run_final_eval_worker
from weiss_rl.eval.targeted_confirm import jobs as _jobs
from weiss_rl.eval.targeted_confirm import plan as _plan
from weiss_rl.eval.targeted_confirm import summary as _summary
from weiss_rl.eval.targeted_confirm.summary import targeted_worker_summary_from_result


def _targeted_worker(job: dict[str, Any]) -> dict[str, Any]:
    result = run_final_eval_worker(job)
    return targeted_worker_summary_from_result(result)


def _run_targeted_jobs(plan: _jobs.TargetedConfirmPlan) -> dict[str, dict[str, Any]]:
    args = plan.args
    results_by_opp: dict[str, dict[str, Any]] = {}
    if int(args.workers) <= 1:
        for job in plan.jobs:
            try:
                result = _targeted_worker(job)
            except Exception:
                traceback.print_exc()
                raise
            opponent = str(job["opponent_policy_id"])
            results_by_opp[opponent] = result
            _print_targeted_progress(
                completed=len(results_by_opp),
                total=len(plan.jobs),
                focal_policy_id=str(args.focal_policy_id),
                opponent=opponent,
                result=result,
            )
        return results_by_opp

    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_targeted_worker, job): job for job in plan.jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception:
                traceback.print_exc()
                raise
            opponent = str(job["opponent_policy_id"])
            results_by_opp[opponent] = result
            _print_targeted_progress(
                completed=len(results_by_opp),
                total=len(plan.jobs),
                focal_policy_id=str(args.focal_policy_id),
                opponent=opponent,
                result=result,
            )
    return results_by_opp


def _print_targeted_progress(
    *,
    completed: int,
    total: int,
    focal_policy_id: str,
    opponent: str,
    result: dict[str, Any],
) -> None:
    print(
        f"done {completed}/{total} {focal_policy_id} vs {opponent} "
        f"mean={result.get('mean')} wins={result.get('wins')}/{result.get('games')}",
        flush=True,
    )


def main() -> None:
    args = _plan.parse_args()
    plan = _jobs.prepare_targeted_confirm_plan(args)
    started = time.time()
    print(
        f"targeted confirm start: focal={args.focal_policy_id} rows={len(plan.opponents)} "
        f"paired_seeds={len(plan.paired_seeds)} workers={args.workers}",
        flush=True,
    )
    results_by_opp = _run_targeted_jobs(plan)
    summary_path, summary = _summary.write_targeted_confirm_summary(
        plan=plan,
        results_by_opp=results_by_opp,
        started_unix=started,
    )
    print(f"summary {summary_path}", flush=True)
    print(json.dumps(summary["overall"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
