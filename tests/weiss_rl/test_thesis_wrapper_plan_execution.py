from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_thesis_wrapper_plan_execution_runs_selected_commands_in_order(tmp_path: Path) -> None:
    from dataclasses import replace

    from weiss_rl.workflows.thesis_wrapper_support.plan_execution import (
        run_thesis_wrapper_commands,
        thesis_wrapper_commands_for_plan,
    )
    from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperPlan, ThesisWrapperResult

    plan = ThesisWrapperPlan(
        repo_root=tmp_path / "repo",
        python_exe="python.exe",
        run_label="demo_run",
        run_dir=tmp_path / "repo" / "runs" / "demo_run",
        stack_config=tmp_path / "repo" / "configs" / "train.yaml",
        eval_stack_config=tmp_path / "repo" / "configs" / "eval.yaml",
        preset="standard",
        eval_preset="standard-thesis-eval",
        train_command=["python.exe", "-m", "weiss_rl.training.train_entrypoint"],
        eval_command=["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"],
        compare_command=["python.exe", "-m", "weiss_rl.workflows.compare_runs.compare_runs_entrypoint"],
        b1_baseline_run_dir=None,
        dry_run=True,
    )
    observed: list[list[str]] = []

    def fake_run_step(*, command: list[str], cwd: Path, dry_run: bool) -> dict[str, object]:
        assert cwd == plan.repo_root
        assert dry_run is True
        observed.append(command)
        return {"command": command, "status": "planned"}

    result = run_thesis_wrapper_commands(plan=plan, run_step_fn=fake_run_step)
    skipped_plan = replace(plan, eval_command=None, compare_command=None)

    assert isinstance(result, ThesisWrapperResult)
    assert result.failed is False
    assert result.status == "planned"
    assert observed == [plan.train_command, plan.eval_command, plan.compare_command]
    assert thesis_wrapper_commands_for_plan(plan) == [plan.train_command, plan.eval_command, plan.compare_command]
    assert thesis_wrapper_commands_for_plan(skipped_plan) == [plan.train_command]


def test_thesis_wrapper_plan_execution_records_failure_without_running_remaining_commands(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_support.plan_execution import run_thesis_wrapper_commands
    from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperPlan

    plan = ThesisWrapperPlan(
        repo_root=tmp_path / "repo",
        python_exe="python.exe",
        run_label="failed_run",
        run_dir=tmp_path / "repo" / "runs" / "failed_run",
        stack_config=tmp_path / "repo" / "configs" / "train.yaml",
        eval_stack_config=tmp_path / "repo" / "configs" / "eval.yaml",
        preset="standard",
        eval_preset="standard-thesis-eval",
        train_command=["python.exe", "-m", "weiss_rl.training.train_entrypoint"],
        eval_command=["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"],
        compare_command=["python.exe", "-m", "weiss_rl.workflows.compare_runs.compare_runs_entrypoint"],
        b1_baseline_run_dir=None,
        dry_run=False,
    )
    observed: list[list[str]] = []

    def fake_run_step(*, command: list[str], cwd: Path, dry_run: bool) -> dict[str, object]:
        observed.append(command)
        if command == plan.eval_command:
            raise subprocess.CalledProcessError(returncode=17, cmd=command)
        return {"command": command, "cwd": cwd.as_posix(), "status": "completed"}

    result = run_thesis_wrapper_commands(plan=plan, run_step_fn=fake_run_step)

    assert result.failed is True
    assert result.status == "failed"
    assert result.steps == [{"command": plan.train_command, "cwd": plan.repo_root.as_posix(), "status": "completed"}]
    assert observed == [plan.train_command, plan.eval_command]


def test_thesis_wrapper_plan_runner_records_failed_step_and_summary(tmp_path: Path, monkeypatch) -> None:
    import weiss_rl.workflows.thesis_wrapper_support.plan as thesis_wrapper_plan
    from weiss_rl.workflows.thesis_wrapper_support.plan import (
        ThesisWrapperPlan,
        run_thesis_wrapper_plan,
        write_thesis_wrapper_summary,
    )

    repo_root = tmp_path / "repo"
    run_dir = repo_root / "runs" / "failed_run"
    plan = ThesisWrapperPlan(
        repo_root=repo_root,
        python_exe="python.exe",
        run_label="failed_run",
        run_dir=run_dir,
        stack_config=repo_root / "configs" / "train.yaml",
        eval_stack_config=repo_root / "configs" / "eval.yaml",
        preset="standard",
        eval_preset="standard-thesis-eval",
        train_command=["python.exe", "-m", "weiss_rl.training.train_entrypoint"],
        eval_command=["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"],
        compare_command=None,
        b1_baseline_run_dir=None,
        dry_run=False,
    )

    def fake_run_step(*, command: list[str], cwd: Path, dry_run: bool) -> dict[str, object]:
        assert cwd == repo_root
        assert dry_run is False
        if "weiss_rl.training.train_entrypoint" in command:
            raise subprocess.CalledProcessError(returncode=17, cmd=command)
        return {"command": command, "cwd": cwd.as_posix(), "status": "completed"}

    monkeypatch.setattr(thesis_wrapper_plan, "_run_step", fake_run_step)

    result = run_thesis_wrapper_plan(plan)
    summary_path = write_thesis_wrapper_summary(result)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result.failed is True
    assert result.status == "failed"
    assert result.steps == []
    assert summary_path == run_dir / "thesis_run_summary.json"
    assert payload["status"] == "failed"
    assert payload["steps"] == []
