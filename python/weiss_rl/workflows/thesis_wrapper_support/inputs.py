from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ThesisWrapperInputs:
    run_label: str
    stack_config: Path | None
    eval_stack_config: Path | None
    preset: str
    eval_preset: str
    num_envs: int
    unroll_length: int
    max_updates: int
    runtime_mode: str
    profile: str
    device: str
    seed: int | None
    resume_run_dir: Path | None
    resume_from: str
    b1_baseline_run_dir: Path | None
    compare_run_dirs: tuple[str, ...]
    compare_launch_group_summary: Path | None
    compare_out_dir: Path | None
    train_args: tuple[str, ...]
    eval_args: tuple[str, ...]
    compare_args: tuple[str, ...]
    skip_eval: bool
    skip_compare: bool
    dry_run: bool


def thesis_wrapper_inputs_from_args(args: Any) -> ThesisWrapperInputs:
    return ThesisWrapperInputs(
        run_label=str(args.run_label),
        stack_config=args.stack_config,
        eval_stack_config=args.eval_stack_config,
        preset=str(args.preset),
        eval_preset=str(args.eval_preset),
        num_envs=int(args.num_envs),
        unroll_length=int(args.unroll_length),
        max_updates=int(args.max_updates),
        runtime_mode=str(args.runtime_mode),
        profile=str(args.profile),
        device=str(args.device),
        seed=args.seed,
        resume_run_dir=args.resume_run_dir,
        resume_from=str(args.resume_from),
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        compare_run_dirs=tuple(str(path) for path in args.compare_run_dir or ()),
        compare_launch_group_summary=args.compare_launch_group_summary,
        compare_out_dir=args.compare_out_dir,
        train_args=tuple(str(extra) for extra in args.train_arg or ()),
        eval_args=tuple(str(extra) for extra in args.eval_arg or ()),
        compare_args=tuple(str(extra) for extra in args.compare_arg or ()),
        skip_eval=bool(args.skip_eval),
        skip_compare=bool(args.skip_compare),
        dry_run=bool(args.dry_run),
    )
