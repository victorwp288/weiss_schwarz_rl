from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from weiss_rl.job_telemetry import ProcessTreeTelemetrySampler, build_benchmark_summary, query_gpu_metrics, write_telemetry_sample


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train.py with whole-job process + GPU telemetry sampling.")
    parser.add_argument("--run-label", required=True, help="Stable run label. Must match the child train.py label.")
    parser.add_argument("--stack-config", required=True, help="Stack config path forwarded to train.py.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--unroll-length", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--runtime-mode", default=None)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--python-executable", default=None, help="Optional python executable. Defaults to sys.executable.")
    parser.add_argument(
        "--override",
        "--config-override",
        dest="config_override",
        action="append",
        default=None,
        help="Forwarded to train.py. Repeat per override.",
    )
    parser.add_argument("--train-arg", action="append", default=None, help="Extra raw train.py token. Repeat per token.")
    args = parser.parse_args()

    repo_root = _repo_root()
    run_dir = repo_root / "runs" / str(args.run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    telemetry_path = run_dir / "job_telemetry.jsonl"
    summary_path = run_dir / "job_telemetry_summary.json"

    command = [args.python_executable or sys.executable, "python/scripts/train.py", "--stack-config", args.stack_config, "--run-label", args.run_label]
    if args.device:
        command.extend(["--device", args.device])
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.num_envs is not None:
        command.extend(["--num-envs", str(args.num_envs)])
    if args.unroll_length is not None:
        command.extend(["--unroll-length", str(args.unroll_length)])
    if args.max_updates is not None:
        command.extend(["--max-updates", str(args.max_updates)])
    if args.runtime_mode:
        command.extend(["--runtime-mode", args.runtime_mode])
    for override in args.config_override or []:
        command.extend(["--override", override])
    command.extend(args.train_arg or [])

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(command, cwd=repo_root, stdout=stdout_handle, stderr=stderr_handle)
        sampler = ProcessTreeTelemetrySampler()
        try:
            while process.poll() is None:
                try:
                    process_payload = sampler.sample(process.pid)
                except Exception:
                    process_payload = None
                write_telemetry_sample(
                    telemetry_path,
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                        "root_pid": int(process.pid),
                        "process": process_payload,
                        "gpu": query_gpu_metrics(),
                    },
                )
                time.sleep(max(float(args.sample_interval_seconds), 0.25))
        finally:
            write_telemetry_sample(
                telemetry_path,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "root_pid": int(process.pid),
                    "process": None,
                    "gpu": query_gpu_metrics(),
                },
            )
            exit_code = int(process.wait())

    summary = build_benchmark_summary(run_dir=run_dir, telemetry_path=telemetry_path)
    summary["command"] = command
    summary["exit_code"] = exit_code
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Profiled training run finished: exit_code={exit_code} run_dir={run_dir}")
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
