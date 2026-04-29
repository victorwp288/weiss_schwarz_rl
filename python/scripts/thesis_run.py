from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PRESET_PATHS = {
    "thesis-model-auto-gpu": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-eval-auto-gpu": Path("configs/main_eval.yaml"),
    "thesis-model-server-train": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-server-train-b1anchored": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-server-train-b1anchored-refb1strong-lowlr": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-server-train-b1anchored-benchmark": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-server-train-b1anchored-benchmark-localpromo": Path("configs/main_impala_league_server.yaml"),
    "thesis-model-server-train-b1anchored-benchmark-selfplay-localpromo": Path(
        "configs/main_impala_league_server.yaml"
    ),
    "thesis-model-server-train-b1anchored-benchmark-selfplay-bckl-localpromo": Path(
        "configs/main_impala_league_server.yaml"
    ),
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-localpromo": Path(
        "configs/main_impala_league_server.yaml"
    ),
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-evalguard-localpromo": Path(
        "configs/main_impala_league_server.yaml"
    ),
    "thesis-model-server-train-b1anchored-benchmark-modelbridge-localpromo": Path(
        "configs/main_impala_league_server.yaml"
    ),
    "b1-anchor-fullsize-warmup": Path("configs/baselines/noleague_fullsize_warmup.yaml"),
    "b1-anchor-fullsize-lowlr-continuation": Path("configs/baselines/noleague_fullsize_lowlr_continuation.yaml"),
    "b1-anchor-benchmark": Path("configs/baselines/noleague_benchmark.yaml"),
    "b1-anchor-benchmark-warmup": Path("configs/baselines/noleague_benchmark_warmup.yaml"),
    "b1-anchor-benchmark-lowlr-continuation": Path("configs/baselines/noleague_benchmark_lowlr_continuation.yaml"),
    "b1-anchor-benchmark-eval-auto-gpu": Path("configs/baselines/noleague_benchmark_eval.yaml"),
    "thesis-model-multideck": Path("configs/ablations/multideck.yaml"),
    "thesis-model-multideck-eval-auto-gpu": Path("configs/ablations/multideck_eval.yaml"),
    "ablate-teacher-fade": Path("configs/ablations/teacher_fade.yaml"),
    "ablate-teacher-fade-eval-auto-gpu": Path("configs/ablations/teacher_fade_eval.yaml"),
    "ablate-no-tactical-bias": Path("configs/ablations/no_tactical_bias.yaml"),
    "ablate-no-tactical-bias-eval-auto-gpu": Path("configs/ablations/no_tactical_bias_eval.yaml"),
    "ablate-teacher-fade-no-tactical-bias": Path("configs/ablations/teacher_fade_no_tactical_bias.yaml"),
    "ablate-teacher-fade-no-tactical-bias-eval-auto-gpu": Path(
        "configs/ablations/teacher_fade_no_tactical_bias_eval.yaml"
    ),
    "ablate-no-b1-cutoff": Path("configs/ablations/no_b1_cutoff.yaml"),
    "ablate-no-b1-cutoff-eval-auto-gpu": Path("configs/ablations/no_b1_cutoff_eval.yaml"),
    "ablate-reward-shaping": Path("configs/ablations/reward_shaping.yaml"),
    "ablate-reward-shaping-eval-auto-gpu": Path("configs/ablations/reward_shaping_eval.yaml"),
}
_DEFAULT_EVAL_PRESET = "thesis-model-eval-auto-gpu"
_DEFAULT_EVAL_PRESET_OVERRIDES = {
    "thesis-model-auto-gpu": "thesis-model-eval-auto-gpu",
    "thesis-model-server-train": "thesis-model-eval-auto-gpu",
    "thesis-model-server-train-b1anchored": "thesis-model-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-refb1strong-lowlr": "thesis-model-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-bckl-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-evalguard-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-server-train-b1anchored-benchmark-modelbridge-localpromo": "b1-anchor-benchmark-eval-auto-gpu",
    "b1-anchor-fullsize-warmup": "thesis-model-eval-auto-gpu",
    "b1-anchor-fullsize-lowlr-continuation": "thesis-model-eval-auto-gpu",
    "b1-anchor-benchmark": "b1-anchor-benchmark-eval-auto-gpu",
    "b1-anchor-benchmark-warmup": "b1-anchor-benchmark-eval-auto-gpu",
    "b1-anchor-benchmark-lowlr-continuation": "b1-anchor-benchmark-eval-auto-gpu",
    "thesis-model-multideck": "thesis-model-multideck-eval-auto-gpu",
    "ablate-teacher-fade": "ablate-teacher-fade-eval-auto-gpu",
    "ablate-no-tactical-bias": "ablate-no-tactical-bias-eval-auto-gpu",
    "ablate-teacher-fade-no-tactical-bias": "ablate-teacher-fade-no-tactical-bias-eval-auto-gpu",
    "ablate-no-b1-cutoff": "ablate-no-b1-cutoff-eval-auto-gpu",
    "ablate-reward-shaping": "ablate-reward-shaping-eval-auto-gpu",
}
_SERVER_TRAIN_PRESETS = {
    "thesis-model-auto-gpu",
    "thesis-model-server-train",
    "thesis-model-server-train-b1anchored",
    "thesis-model-server-train-b1anchored-refb1strong-lowlr",
    "thesis-model-server-train-b1anchored-benchmark",
    "thesis-model-server-train-b1anchored-benchmark-localpromo",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-localpromo",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-bckl-localpromo",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-localpromo",
    "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-evalguard-localpromo",
    "thesis-model-server-train-b1anchored-benchmark-modelbridge-localpromo",
}


def _argv_has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(token.startswith(option + "=") for token in argv)


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


def _train_entrypoint_command(*, python_exe: str, torchrun_nproc: int) -> list[str]:
    if int(torchrun_nproc) <= 0:
        return [python_exe, "python/scripts/train.py"]
    return [
        python_exe,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(int(torchrun_nproc)),
        "python/scripts/train.py",
    ]


def _resolve_cli_path(*, repo_root: Path, path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _resolve_stack_config(*, repo_root: Path, stack_config: Path | None, preset: str) -> Path:
    if stack_config is not None:
        return _resolve_cli_path(repo_root=repo_root, path=stack_config)
    return (repo_root / _PRESET_PATHS[preset]).resolve()


def _resolve_eval_stack_config(
    *,
    repo_root: Path,
    eval_stack_config: Path | None,
    train_stack_config: Path | None,
    eval_preset: str,
) -> Path:
    if eval_stack_config is not None:
        return _resolve_cli_path(repo_root=repo_root, path=eval_stack_config)
    if train_stack_config is not None and not eval_preset:
        return _resolve_cli_path(repo_root=repo_root, path=train_stack_config)
    return (repo_root / _PRESET_PATHS[eval_preset]).resolve()


def _default_eval_preset_for_preset(preset: str) -> str:
    return _DEFAULT_EVAL_PRESET_OVERRIDES.get(preset, _DEFAULT_EVAL_PRESET)


def _should_run_compare(
    *,
    compare_run_dirs: list[str] | None,
    compare_launch_group_summary: Path | None,
) -> bool:
    return bool(compare_run_dirs) or compare_launch_group_summary is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin wrapper for canonical thesis train/eval/compare runs")
    parser.add_argument("--stack-config", type=Path, default=None)
    parser.add_argument("--eval-stack-config", type=Path, default=None)
    parser.add_argument("--preset", choices=tuple(_PRESET_PATHS), default="thesis-model-auto-gpu")
    parser.add_argument("--eval-preset", choices=tuple(_PRESET_PATHS), default="")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--unroll-length", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--max-wall-clock-minutes", type=float, default=None)
    parser.add_argument("--runtime-mode", type=str, default="train_ordered")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--autoscale-dry-run", action="store_true")
    parser.add_argument("--hardware-profile", type=str, default="")
    parser.add_argument("--torchrun-nproc", type=int, default=0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-backend", type=str, default="")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--seed-snapshot-run-dir", type=Path, default=None)
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
    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)

    repo_root = Path(__file__).resolve().parents[2] if args.repo_root is None else Path(args.repo_root).resolve()
    if args.list_presets:
        for name, path in _PRESET_PATHS.items():
            print(f"{name}: {(repo_root / path).as_posix()}")
        return
    if not str(args.run_label).strip():
        parser.error("--run-label is required unless --list-presets is used")
    server_defaults_applied = False
    if args.stack_config is None and str(args.preset) in _SERVER_TRAIN_PRESETS:
        server_defaults_applied = True
        if not bool(args.autoscale or args.autoscale_dry_run):
            args.autoscale = True
        if not str(args.hardware_profile).strip():
            args.hardware_profile = "local"
        if not _argv_has_option(raw_argv, "--runtime-mode"):
            args.runtime_mode = "train_async_fast"
        if not _argv_has_option(raw_argv, "--unroll-length"):
            args.unroll_length = 64
        if not _argv_has_option(raw_argv, "--max-updates"):
            args.max_updates = 400
    num_envs_explicit = _argv_has_option(raw_argv, "--num-envs")
    python_exe = sys.executable
    run_dir = repo_root / "runs" / args.run_label
    stack_config = _resolve_stack_config(repo_root=repo_root, stack_config=args.stack_config, preset=str(args.preset))
    eval_preset = str(args.eval_preset).strip()
    if not eval_preset and args.stack_config is None:
        eval_preset = _default_eval_preset_for_preset(str(args.preset))
    eval_stack_config = _resolve_eval_stack_config(
        repo_root=repo_root,
        eval_stack_config=args.eval_stack_config,
        train_stack_config=args.stack_config,
        eval_preset=eval_preset,
    )
    steps: list[dict[str, Any]] = []

    train_command = [
        *_train_entrypoint_command(python_exe=python_exe, torchrun_nproc=int(args.torchrun_nproc)),
        "--stack-config",
        str(stack_config),
        "--run-label",
        args.run_label,
    ]
    if not (bool(args.autoscale or args.autoscale_dry_run) and not num_envs_explicit):
        train_command.extend(["--num-envs", str(args.num_envs)])
    train_command.extend(
        [
            "--unroll-length",
            str(args.unroll_length),
            "--max-updates",
            str(args.max_updates),
            "--runtime-mode",
            str(args.runtime_mode),
        ]
    )
    if args.max_wall_clock_minutes is not None:
        train_command.extend(["--max-wall-clock-minutes", str(args.max_wall_clock_minutes)])
    if args.profile:
        train_command.extend(["--profile", str(args.profile)])
    if args.device:
        train_command.extend(["--device", str(args.device)])
    if args.autoscale:
        train_command.append("--autoscale")
    if args.autoscale_dry_run:
        train_command.append("--autoscale-dry-run")
    if args.hardware_profile:
        train_command.extend(["--hardware-profile", str(args.hardware_profile)])
    if args.ddp or int(args.torchrun_nproc) > 0:
        train_command.append("--ddp")
    if args.ddp_backend:
        train_command.extend(["--ddp-backend", str(args.ddp_backend)])
    if int(args.ddp_timeout_seconds) > 0:
        train_command.extend(["--ddp-timeout-seconds", str(args.ddp_timeout_seconds)])
    if args.seed is not None:
        train_command.extend(["--seed", str(args.seed)])
    if args.resume_run_dir is not None:
        train_command.extend(["--resume-run-dir", str(args.resume_run_dir)])
    if args.resume_from:
        train_command.extend(["--resume-from", str(args.resume_from)])
    if args.b1_baseline_run_dir is not None:
        train_command.extend(["--b1-baseline-run-dir", str(args.b1_baseline_run_dir)])
    if args.seed_snapshot_run_dir is not None:
        train_command.extend(["--seed-snapshot-run-dir", str(args.seed_snapshot_run_dir)])
    for extra in args.train_arg or []:
        train_command.append(str(extra))
    failed = False
    try:
        steps.append(_run_step(command=train_command, cwd=repo_root, dry_run=bool(args.dry_run)))

        if not args.skip_eval and not args.autoscale_dry_run:
            eval_command = [
                python_exe,
                "python/scripts/eval.py",
                "--stack-config",
                str(eval_stack_config),
                "--run-dir",
                str(run_dir),
            ]
            if args.b1_baseline_run_dir is not None:
                eval_command.extend(["--b1-baseline-run-dir", str(args.b1_baseline_run_dir)])
            for extra in args.eval_arg or []:
                eval_command.append(str(extra))
            steps.append(_run_step(command=eval_command, cwd=repo_root, dry_run=bool(args.dry_run)))

        should_run_compare = _should_run_compare(
            compare_run_dirs=args.compare_run_dir,
            compare_launch_group_summary=args.compare_launch_group_summary,
        )
        if not args.skip_compare and not args.autoscale_dry_run and should_run_compare:
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
        "max_wall_clock_minutes": None if args.max_wall_clock_minutes is None else float(args.max_wall_clock_minutes),
        "autoscale": bool(args.autoscale),
        "autoscale_dry_run": bool(args.autoscale_dry_run),
        "server_defaults_applied": bool(server_defaults_applied),
        "hardware_profile": str(args.hardware_profile),
        "torchrun_nproc": int(args.torchrun_nproc),
        "ddp": bool(args.ddp or int(args.torchrun_nproc) > 0),
        "ddp_backend": str(args.ddp_backend),
        "ddp_timeout_seconds": int(args.ddp_timeout_seconds),
        "b1_baseline_run_dir": (
            None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.resolve().as_posix()
        ),
        "seed_snapshot_run_dir": (
            None if args.seed_snapshot_run_dir is None else args.seed_snapshot_run_dir.resolve().as_posix()
        ),
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
