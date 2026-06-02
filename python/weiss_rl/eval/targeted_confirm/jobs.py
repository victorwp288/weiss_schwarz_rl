from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import parse_seed_file
from weiss_rl.config import load_stack_config
from weiss_rl.eval.targeted_confirm.opponents import validate_targeted_confirm_request
from weiss_rl.eval.targeted_confirm.summary import god_search_payload_from_args


@dataclass(frozen=True)
class TargetedConfirmPlan:
    args: argparse.Namespace
    opponents: list[str]
    paired_seeds: list[int]
    seed_file_path: Path
    seed_source: str
    out_dir: Path
    jobs: list[dict[str, Any]]
    eval_sampling_algorithm: str | None
    model_sampling_temperature: float | None


def resolve_paired_seed_file(args: argparse.Namespace, stack: Any) -> tuple[Path, str]:
    explicit_path = getattr(args, "paired_seed_file", None)
    if explicit_path is not None:
        return Path(explicit_path).resolve(), "explicit"
    seed_set = str(getattr(args, "seed_set", "report_eval"))
    if seed_set not in stack.seed_sets:
        raise KeyError(f"seed set not found in stack config: {seed_set}")
    return Path(stack.seed_sets[seed_set]), seed_set


def targeted_eval_job(
    *,
    args: argparse.Namespace,
    paired_seeds: list[int],
    opponent_index: int,
    opponent: str,
    output_dir: Path,
) -> dict[str, Any]:
    job: dict[str, Any] = {
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
    god_search = god_search_payload_from_args(args)
    if god_search is not None:
        job["god_search"] = god_search
    return job


def build_targeted_confirm_jobs(
    *,
    args: argparse.Namespace,
    paired_seeds: list[int],
    opponents: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    return [
        targeted_eval_job(
            args=args,
            paired_seeds=paired_seeds,
            opponent_index=idx,
            opponent=opponent,
            output_dir=output_dir,
        )
        for idx, opponent in enumerate(opponents, start=1)
    ]


def prepare_targeted_confirm_plan(
    args: argparse.Namespace,
    *,
    opponents: list[str] | None = None,
) -> TargetedConfirmPlan:
    selected_opponents = validate_targeted_confirm_request(args) if opponents is None else opponents
    stack = load_stack_config(args.stack_config)
    seed_file_path, seed_source = resolve_paired_seed_file(args, stack)
    paired_seeds = parse_seed_file(seed_file_path)[: int(args.paired_seeds)]
    if len(paired_seeds) < int(args.paired_seeds):
        raise RuntimeError(f"requested {args.paired_seeds} paired seeds, found {len(paired_seeds)} in {seed_file_path}")

    out_dir = args.run_dir / "eval" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = stack.config.evaluation
    jobs = build_targeted_confirm_jobs(
        args=args,
        paired_seeds=paired_seeds,
        opponents=selected_opponents,
        output_dir=out_dir,
    )
    return TargetedConfirmPlan(
        args=args,
        opponents=selected_opponents,
        paired_seeds=paired_seeds,
        seed_file_path=seed_file_path,
        seed_source=seed_source,
        out_dir=out_dir,
        jobs=jobs,
        eval_sampling_algorithm=None if evaluation is None else str(getattr(evaluation, "eval_sampling_algorithm", "")),
        model_sampling_temperature=None
        if evaluation is None
        else float(getattr(evaluation, "model_sampling_temperature", 1.0)),
    )


__all__ = [
    "TargetedConfirmPlan",
    "build_targeted_confirm_jobs",
    "prepare_targeted_confirm_plan",
    "resolve_paired_seed_file",
    "targeted_eval_job",
]
