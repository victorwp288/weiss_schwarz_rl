"""Single-node experiment launcher with GPU-aware scheduling."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LaunchJob:
    job_id: str
    stack_config: str
    seed: int
    device: str
    run_label: str
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    group_label: str
    jobs: tuple[LaunchJob, ...]
    max_parallel_jobs: int


def resolve_devices(*, requested_devices: list[str] | None, cuda_available: bool, cuda_count: int) -> tuple[str, ...]:
    if requested_devices:
        devices = tuple(str(device).strip() for device in requested_devices if str(device).strip())
        if not devices:
            raise ValueError("requested_devices must contain at least one non-empty entry")
        return devices
    if cuda_available and cuda_count > 0:
        return tuple(f"cuda:{index}" for index in range(cuda_count))
    return ("cpu",)


def build_launch_plan(
    *,
    group_label: str,
    stack_configs: list[str],
    seeds: list[int],
    devices: tuple[str, ...],
    run_label_prefix: str | None = None,
    extra_args: list[str] | None = None,
) -> LaunchPlan:
    if not stack_configs:
        raise ValueError("stack_configs must contain at least one entry")
    if not seeds:
        raise ValueError("seeds must contain at least one entry")
    if not devices:
        raise ValueError("devices must contain at least one entry")

    jobs: list[LaunchJob] = []
    base_prefix = run_label_prefix or group_label
    extra_args_tuple = tuple(extra_args or ())
    job_index = 0
    for stack_config in stack_configs:
        stack_stem = Path(stack_config).stem or "stack"
        for seed in seeds:
            device = devices[job_index % len(devices)]
            run_label = f"{base_prefix}_{stack_stem}_seed{int(seed)}"
            jobs.append(
                LaunchJob(
                    job_id=f"job_{job_index:03d}",
                    stack_config=str(Path(stack_config)),
                    seed=int(seed),
                    device=device,
                    run_label=run_label,
                    extra_args=extra_args_tuple,
                )
            )
            job_index += 1
    return LaunchPlan(group_label=group_label, jobs=tuple(jobs), max_parallel_jobs=len(devices))


def execute_launch_plan(
    *,
    repo_root: Path,
    plan: LaunchPlan,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    runs_root = repo_root / "runs"
    group_dir = runs_root / "launch_groups" / plan.group_label
    group_dir.mkdir(parents=True, exist_ok=True)
    summary_path = group_dir / "summary.json"
    train_script = repo_root / "weiss_schwarz_rl" / "python" / "scripts" / "train.py"
    python_cmd = python_executable or sys.executable

    summary: dict[str, Any] = {
        "kind": "single_node_launch_group_v1",
        "group_label": plan.group_label,
        "repo_root": repo_root.as_posix(),
        "dry_run": bool(dry_run),
        "max_parallel_jobs": int(plan.max_parallel_jobs),
        "jobs": [],
    }

    active: list[tuple[subprocess.Popen[str], LaunchJob, float]] = []
    pending = list(plan.jobs)
    while pending or active:
        while pending and len(active) < int(plan.max_parallel_jobs):
            job = pending.pop(0)
            command = [
                python_cmd,
                str(train_script),
                "--stack-config",
                job.stack_config,
                "--seed",
                str(job.seed),
                "--device",
                job.device,
                "--run-label",
                job.run_label,
                *job.extra_args,
            ]
            started_at = time.time()
            summary["jobs"].append(
                {
                    **asdict(job),
                    "command": command,
                    "expected_run_dir": (repo_root / "weiss_schwarz_rl" / "runs" / job.run_label).as_posix(),
                    "status": "planned" if dry_run else "running",
                    "started_at_unix": started_at,
                }
            )
            if dry_run:
                continue
            active.append(
                (
                    subprocess.Popen(
                        command,
                        cwd=repo_root / "weiss_schwarz_rl",
                        text=True,
                    ),
                    job,
                    started_at,
                )
            )

        if dry_run:
            break

        time.sleep(0.1)
        still_active: list[tuple[subprocess.Popen[str], LaunchJob, float]] = []
        for process, job, started_at in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, job, started_at))
                continue
            _update_job_summary(
                summary,
                job_id=job.job_id,
                status="completed" if return_code == 0 else "failed",
                return_code=int(return_code),
                finished_at=time.time(),
            )
        active = still_active

    if dry_run:
        for job in plan.jobs:
            _update_job_summary(summary, job_id=job.job_id, status="planned", return_code=None, finished_at=None)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _update_job_summary(
    summary: dict[str, Any],
    *,
    job_id: str,
    status: str,
    return_code: int | None,
    finished_at: float | None,
) -> None:
    for job_payload in summary["jobs"]:
        if str(job_payload.get("job_id")) != job_id:
            continue
        job_payload["status"] = status
        job_payload["return_code"] = return_code
        job_payload["finished_at_unix"] = finished_at
        return
    raise KeyError(f"missing launch summary entry for {job_id}")
