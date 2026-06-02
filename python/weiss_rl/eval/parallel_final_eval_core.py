from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from weiss_rl.eval.final.worker import load_json_object, run_final_eval_worker, worker_output_dir
from weiss_rl.eval.parallel_final_eval_plan import (
    ParallelFinalEvalPlan,
    parse_args,
    prepare_parallel_final_eval_plan,
    write_parallel_final_eval_artifacts,
)

_load_json = load_json_object
_worker_output_dir = worker_output_dir
_worker = run_final_eval_worker


def _run_jobs(plan: ParallelFinalEvalPlan) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=plan.workers) as executor:
        futures = {executor.submit(_worker, job): job for job in plan.jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"completed {len(results)}/{len(plan.jobs)}: {job['focal_policy_id']} vs {job['opponent_policy_id']}",
                flush=True,
            )
    results.sort(key=lambda item: (int(item["focal_index"]), int(item["opponent_index"])))
    return results


def main() -> None:
    args = parse_args()
    plan = prepare_parallel_final_eval_plan(args)
    results = _run_jobs(plan)
    write_parallel_final_eval_artifacts(plan=plan, matchup_results=results)
    print(f"Parallel final_eval summary JSON: {plan.layout.final_eval_summary_json()}")


if __name__ == "__main__":
    main()
