from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import parse_seed_file, require_fixed_python_hash_seed
from weiss_rl.config import load_stack_config
from weiss_rl.eval.final.artifacts import write_final_eval_artifacts
from weiss_rl.eval.final.payload import build_final_eval_payload
from weiss_rl.eval.payoff_folding import PayoffFoldScheme


@dataclass(frozen=True)
class ParallelFinalEvalPlan:
    policy_ids: list[str]
    layout: ArtifactLayout
    paired_seeds: list[int]
    stage1_paired_seeds: int
    max_paired_seeds: int
    bootstrap_samples: int
    workers: int
    seed_file_path: Path
    stop_rules: Any
    scheme: PayoffFoldScheme
    jobs: list[dict[str, Any]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel canonical final eval for independent matchups")
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy-id", action="append", default=[])
    parser.add_argument("--snapshot-registry-json", type=Path, default=None)
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--paired-seed-limit", type=int, default=16)
    parser.add_argument("--stage1-paired-seeds", type=int, default=16)
    parser.add_argument("--max-paired-seeds", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--allow-parallel-workers",
        action="store_true",
        help="Allow workers >1. Parallel simulator eval is experimental and should not be used for checkpoint selection.",
    )
    parser.add_argument("--force-clear", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def validate_parallel_final_eval_request(args: argparse.Namespace) -> list[str]:
    policy_ids = [str(p).strip() for p in args.policy_id if str(p).strip()]
    if not policy_ids:
        raise SystemExit("provide at least one --policy-id")
    if int(args.workers) > 1 and not bool(args.allow_parallel_workers):
        raise SystemExit(
            "final eval is deterministic only with --workers 1; "
            "pass --allow-parallel-workers for exploratory non-selection runs"
        )
    try:
        require_fixed_python_hash_seed("final eval")
    except RuntimeError as err:
        raise SystemExit(str(err)) from err
    return policy_ids


def build_parallel_final_eval_jobs(
    *,
    args: argparse.Namespace,
    policy_ids: list[str],
    paired_seeds: list[int],
    scheme: PayoffFoldScheme,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for focal_index, focal_policy_id in enumerate(policy_ids):
        for opponent_index, opponent_policy_id in enumerate(policy_ids[focal_index:], start=focal_index):
            jobs.append(
                {
                    "stack_config": args.stack_config.as_posix(),
                    "run_dir": args.run_dir.as_posix(),
                    "snapshot_registry_json": None
                    if args.snapshot_registry_json is None
                    else args.snapshot_registry_json.as_posix(),
                    "b1_baseline_run_dir": None
                    if args.b1_baseline_run_dir is None
                    else args.b1_baseline_run_dir.as_posix(),
                    "paired_seeds": paired_seeds,
                    "stage1_paired_seeds": int(args.stage1_paired_seeds),
                    "max_paired_seeds": int(args.max_paired_seeds),
                    "bootstrap_samples": int(args.bootstrap_samples),
                    "scheme": scheme,
                    "focal_index": focal_index,
                    "opponent_index": opponent_index,
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                }
            )
    return jobs


def prepare_parallel_final_eval_plan(
    args: argparse.Namespace,
    *,
    policy_ids: list[str] | None = None,
) -> ParallelFinalEvalPlan:
    selected_policy_ids = validate_parallel_final_eval_request(args) if policy_ids is None else policy_ids
    stack = load_stack_config(args.stack_config)
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    layout = ArtifactLayout.from_run_dir(args.run_dir)
    layout.ensure_directories()
    if args.force_clear and layout.final_eval_dir.exists():
        shutil.rmtree(layout.final_eval_dir)
    layout.final_eval_dir.mkdir(parents=True, exist_ok=True)
    seed_file_path = stack.seed_sets["report_eval"]
    paired_seeds = parse_seed_file(seed_file_path)[: int(args.paired_seed_limit)]
    if len(paired_seeds) < int(args.max_paired_seeds):
        raise ValueError("not enough paired seeds")
    scheme = cast(PayoffFoldScheme, str(evaluation.final_policy_set_selection.folding))
    jobs = build_parallel_final_eval_jobs(
        args=args,
        policy_ids=selected_policy_ids,
        paired_seeds=paired_seeds,
        scheme=scheme,
    )
    return ParallelFinalEvalPlan(
        policy_ids=selected_policy_ids,
        layout=layout,
        paired_seeds=paired_seeds,
        stage1_paired_seeds=int(args.stage1_paired_seeds),
        max_paired_seeds=int(args.max_paired_seeds),
        bootstrap_samples=int(args.bootstrap_samples),
        workers=int(args.workers),
        seed_file_path=seed_file_path,
        stop_rules=evaluation.stop_rules,
        scheme=scheme,
        jobs=jobs,
    )


def write_parallel_final_eval_artifacts(
    *,
    plan: ParallelFinalEvalPlan,
    matchup_results: list[dict[str, Any]],
) -> None:
    payload = build_final_eval_payload(
        output_dir=plan.layout.final_eval_dir,
        policy_ids=plan.policy_ids,
        matchup_results=matchup_results,
        stage1_paired_seeds=plan.stage1_paired_seeds,
        max_paired_seeds=plan.max_paired_seeds,
        paired_seeds=plan.paired_seeds,
        stop_rules=plan.stop_rules,
        scheme=plan.scheme,
        sample_count=plan.bootstrap_samples,
        selection_payload={
            "mode": "explicit_parallel_cli",
            "policy_count": len(plan.policy_ids),
            "workers": plan.workers,
        },
        metadata={"pipeline": {"kind": "parallel_final_eval_v1", "workers": plan.workers}},
        seed_file_path=plan.seed_file_path,
    )
    write_final_eval_artifacts(
        output_dir=plan.layout.final_eval_dir,
        payload=payload,
        matchup_results=matchup_results,
    )
