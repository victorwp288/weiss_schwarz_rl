from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiment_launcher import build_launch_plan, execute_launch_plan, resolve_devices


def test_resolve_devices_falls_back_cleanly() -> None:
    assert resolve_devices(requested_devices=None, cuda_available=False, cuda_count=0) == ("cpu",)
    assert resolve_devices(requested_devices=None, cuda_available=True, cuda_count=2) == ("cuda:0", "cuda:1")
    assert resolve_devices(requested_devices=["cuda:3"], cuda_available=True, cuda_count=4) == ("cuda:3",)


def test_build_launch_plan_round_robins_devices() -> None:
    plan = build_launch_plan(
        group_label="bench_a",
        stack_configs=["C:/repo/configs/local.yaml", "C:/repo/configs/baselines/ppo_lite.yaml"],
        seeds=[1, 2],
        devices=("cuda:0", "cuda:1"),
    )

    assert plan.max_parallel_jobs == 2
    assert [job.device for job in plan.jobs] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert plan.jobs[0].run_label.startswith("bench_a_")
    assert len({job.run_label for job in plan.jobs}) == 4


def test_execute_launch_plan_dry_run_writes_group_summary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "python" / "scripts").mkdir(parents=True)
    (repo_root / "python" / "scripts" / "train.py").write_text("print('stub')\n", encoding="utf-8")

    plan = build_launch_plan(
        group_label="bench_b",
        stack_configs=["C:/repo/configs/local.yaml"],
        seeds=[7, 8],
        devices=("cpu",),
        extra_args=["--max-updates", "2"],
    )
    summary = execute_launch_plan(repo_root=repo_root, plan=plan, dry_run=True)

    summary_path = repo_root / "runs" / "launch_groups" / "bench_b" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["group_label"] == "bench_b"
    assert payload["dry_run"] is True
    assert payload["status"] == "planned"
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["command"][-2:] == ["--max-updates", "2"]
    assert payload["jobs"][0]["expected_run_dir"].startswith((repo_root / "runs" / "bench_b_").as_posix())


def test_build_launch_plan_uses_collision_safe_run_labels_for_same_stem() -> None:
    plan = build_launch_plan(
        group_label="bench_collision",
        stack_configs=["/repo/configs/a/stack.yaml", "/repo/configs/b/stack.yaml"],
        seeds=[7],
        devices=("cpu",),
    )

    assert len(plan.jobs) == 2
    assert plan.jobs[0].run_label != plan.jobs[1].run_label


def test_execute_launch_plan_raises_when_any_job_fails(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "python" / "scripts").mkdir(parents=True)
    (repo_root / "python" / "scripts" / "train.py").write_text("import sys\nsys.exit(2)\n", encoding="utf-8")

    plan = build_launch_plan(
        group_label="bench_fail",
        stack_configs=["/repo/configs/stack.yaml"],
        seeds=[7],
        devices=("cpu",),
    )

    with pytest.raises(RuntimeError, match="failed with 1 job"):
        execute_launch_plan(repo_root=repo_root, plan=plan, dry_run=False)

    payload = json.loads(
        (repo_root / "runs" / "launch_groups" / "bench_fail" / "summary.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["failed_job_count"] == 1
