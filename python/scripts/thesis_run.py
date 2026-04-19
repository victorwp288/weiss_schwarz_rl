from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_PRESET_PATHS = {
    "standard": Path("configs/presets/structured_acceptance_standard.yaml"),
    "standard-auto-gpu": Path("configs/presets/structured_acceptance_standard_auto_gpu.yaml"),
    "standard-thesis-eval": Path("configs/presets/structured_acceptance_standard_thesis_eval.yaml"),
    "standard-multideck": Path("configs/presets/structured_acceptance_standard_multideck.yaml"),
    "ablate-no-tactical-bias": Path("configs/presets/ablations/standard_no_tactical_bias.yaml"),
    "ablate-no-b1-cutoff": Path("configs/presets/ablations/standard_no_b1_cutoff.yaml"),
    "ablate-multideck": Path("configs/presets/ablations/standard_multideck_generalization.yaml"),
}
_DEFAULT_EVAL_PRESET = "standard-thesis-eval"


def _command_display(command: list[str]) -> str:
    return " ".join(command)


def _summary_path(repo_root: Path, *, run_label: str, dry_run: bool) -> Path:
    if dry_run:
        return repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
    return repo_root / "runs" / run_label / "thesis_run_summary.json"


def _run_step(*, command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": command,
        "cwd": cwd.as_posix(),
        "status": "planned" if dry_run else "running",
    }
    print(_command_display(command))
    if dry_run:
        return payload
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        payload["status"] = "failed"
        payload["returncode"] = int(exc.returncode)
        raise
    payload["status"] = "completed"
    payload["returncode"] = 0
    return payload


def _resolve_stack_config(*, repo_root: Path, stack_config: Path | None, preset: str) -> Path:
    if stack_config is not None:
        return Path(stack_config).resolve()
    return (repo_root / _PRESET_PATHS[preset]).resolve()


def _resolve_eval_stack_config(
    *,
    repo_root: Path,
    eval_stack_config: Path | None,
    train_stack_config: Path | None,
    eval_preset: str,
) -> Path:
    if eval_stack_config is not None:
        return Path(eval_stack_config).resolve()
    if train_stack_config is not None and not eval_preset:
        return Path(train_stack_config).resolve()
    return (repo_root / _PRESET_PATHS[eval_preset]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin wrapper for canonical thesis train/eval/compare runs")
    parser.add_argument("--stack-config", type=Path, default=None)
    parser.add_argument("--eval-stack-config", type=Path, default=None)
    parser.add_argument("--preset", choices=tuple(_PRESET_PATHS), default="standard")
    parser.add_argument("--eval-preset", choices=tuple(_PRESET_PATHS), default="")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--unroll-length", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--runtime-mode", type=str, default="train_ordered")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--compare-run-dir", action="append", default=None)
    parser.add_argument("--compare-launch-group-summary", type=Path, default=None)
    parser.add_argument("--compare-out-dir", type=Path, default=None)
    parser.add_argument("--train-arg", action="append", default=None)
    parser.add_argument("--eval-arg", action="append", default=None)
    parser.add_argument("--compare-arg", action="append", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2] if args.repo_root is None else Path(args.repo_root).resolve()
    if args.list_presets:
        for name, path in _PRESET_PATHS.items():
            print(f"{name}: {(repo_root / path).as_posix()}")
        return
    if not str(args.run_label).strip():
        parser.error("--run-label is required unless --list-presets is used")
    python_exe = sys.executable
    run_dir = repo_root / "runs" / args.run_label
    stack_config = _resolve_stack_config(repo_root=repo_root, stack_config=args.stack_config, preset=str(args.preset))
    eval_preset = str(args.eval_preset).strip()
    if not eval_preset and args.stack_config is None:
        eval_preset = _DEFAULT_EVAL_PRESET
    eval_stack_config = _resolve_eval_stack_config(
        repo_root=repo_root,
        eval_stack_config=args.eval_stack_config,
        train_stack_config=args.stack_config,
        eval_preset=eval_preset,
    )
    steps: list[dict[str, Any]] = []

    train_command = [
        python_exe,
        "python/scripts/train.py",
        "--stack-config",
        str(stack_config),
        "--run-label",
        args.run_label,
        "--num-envs",
        str(args.num_envs),
        "--unroll-length",
        str(args.unroll_length),
        "--max-updates",
        str(args.max_updates),
        "--runtime-mode",
        str(args.runtime_mode),
    ]
    if args.profile:
        train_command.extend(["--profile", str(args.profile)])
    if args.device:
        train_command.extend(["--device", str(args.device)])
    if args.seed is not None:
        train_command.extend(["--seed", str(args.seed)])
    if args.resume_run_dir is not None:
        train_command.extend(["--resume-run-dir", str(args.resume_run_dir)])
    if args.resume_from:
        train_command.extend(["--resume-from", str(args.resume_from)])
    for extra in args.train_arg or []:
        train_command.append(str(extra))
    failed = False
    try:
        steps.append(_run_step(command=train_command, cwd=repo_root, dry_run=bool(args.dry_run)))

        if not args.skip_eval:
            eval_command = [
                python_exe,
                "python/scripts/eval.py",
                "--stack-config",
                str(eval_stack_config),
                "--run-dir",
                str(run_dir),
            ]
            for extra in args.eval_arg or []:
                eval_command.append(str(extra))
            steps.append(_run_step(command=eval_command, cwd=repo_root, dry_run=bool(args.dry_run)))

        if not args.skip_compare:
            compare_command = [
                python_exe,
                "python/scripts/compare_runs.py",
                "--run-dir",
                str(run_dir),
            ]
            for baseline_run_dir in args.compare_run_dir or []:
                compare_command.extend(["--run-dir", str(baseline_run_dir)])
            if args.compare_launch_group_summary is not None:
                compare_command.extend(["--launch-group-summary", str(args.compare_launch_group_summary)])
            if args.compare_out_dir is not None:
                compare_command.extend(["--out-dir", str(args.compare_out_dir)])
            for extra in args.compare_arg or []:
                compare_command.append(str(extra))
            steps.append(_run_step(command=compare_command, cwd=repo_root, dry_run=bool(args.dry_run)))
    except subprocess.CalledProcessError:
        failed = True
    summary_payload = {
        "kind": "thesis_run_wrapper_v1",
        "run_label": args.run_label,
        "run_dir": run_dir.as_posix(),
        "stack_config": stack_config.as_posix(),
        "eval_stack_config": eval_stack_config.as_posix(),
        "preset": str(args.preset),
        "eval_preset": eval_preset,
        "dry_run": bool(args.dry_run),
        "status": "failed" if failed else ("planned" if args.dry_run else "completed"),
        "steps": steps,
    }
    summary_path = _summary_path(repo_root, run_label=args.run_label, dry_run=bool(args.dry_run))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote thesis wrapper summary: {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
