from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.experiment_launcher import build_launch_plan, execute_launch_plan, resolve_devices


def test_resolve_devices_falls_back_cleanly() -> None:
    assert resolve_devices(requested_devices=None, cuda_available=False, cuda_count=0) == ("cpu",)
    assert resolve_devices(requested_devices=None, cuda_available=True, cuda_count=2) == ("cuda:0", "cuda:1")
    assert resolve_devices(requested_devices=["cuda:3"], cuda_available=True, cuda_count=4) == ("cuda:3",)


def test_build_launch_plan_round_robins_devices() -> None:
    plan = build_launch_plan(
        group_label="bench_a",
        stack_configs=["C:/repo/configs/presets/typed_local.yaml", "C:/repo/configs/presets/baselines/ppo_lite.yaml"],
        seeds=[1, 2],
        devices=("cuda:0", "cuda:1"),
    )

    assert plan.max_parallel_jobs == 2
    assert [job.device for job in plan.jobs] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert plan.jobs[0].run_label.startswith("bench_a_")


def test_execute_launch_plan_dry_run_writes_group_summary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "weiss_schwarz_rl" / "python" / "scripts").mkdir(parents=True)
    (repo_root / "weiss_schwarz_rl" / "python" / "scripts" / "train.py").write_text("print('stub')\n", encoding="utf-8")

    plan = build_launch_plan(
        group_label="bench_b",
        stack_configs=["C:/repo/configs/presets/typed_local.yaml"],
        seeds=[7, 8],
        devices=("cpu",),
        extra_args=["--max-updates", "2"],
    )
    summary = execute_launch_plan(repo_root=repo_root, plan=plan, dry_run=True)

    summary_path = repo_root / "runs" / "launch_groups" / "bench_b" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["group_label"] == "bench_b"
    assert payload["dry_run"] is True
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["command"][-2:] == ["--max-updates", "2"]
    assert payload["jobs"][0]["expected_run_dir"].endswith("/weiss_schwarz_rl/runs/bench_b_typed_local_seed7")
