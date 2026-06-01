from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(script_name: str):
    script_path = REPO_ROOT / "python" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_script_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_thesis_run_script_is_thin_workflow_facade() -> None:
    script_module = _load_script_module("thesis_run.py")
    from weiss_rl.workflows import (
        thesis_wrapper,
        thesis_wrapper_cli,
        thesis_wrapper_plan,
        thesis_wrapper_plan_commands,
        thesis_wrapper_plan_execution,
        thesis_wrapper_request,
        thesis_wrapper_state,
        thesis_wrapper_summary,
    )

    assert script_module.main is thesis_wrapper.main
    assert script_module._PRESET_PATHS is thesis_wrapper._PRESET_PATHS
    assert script_module._resolve_stack_config is thesis_wrapper._resolve_stack_config
    assert script_module._resolve_eval_stack_config is thesis_wrapper._resolve_eval_stack_config
    assert script_module.build_thesis_train_command is thesis_wrapper.build_thesis_train_command
    assert script_module.build_thesis_eval_command is thesis_wrapper.build_thesis_eval_command
    assert script_module.build_thesis_compare_command is thesis_wrapper.build_thesis_compare_command
    assert script_module.build_thesis_wrapper_commands is thesis_wrapper.build_thesis_wrapper_commands
    assert script_module.build_thesis_wrapper_parser is thesis_wrapper.build_thesis_wrapper_parser
    assert script_module.ThesisWrapperInputs is thesis_wrapper.ThesisWrapperInputs
    assert script_module.ThesisWrapperCommands is thesis_wrapper.ThesisWrapperCommands
    assert script_module.ThesisWrapperRequest is thesis_wrapper.ThesisWrapperRequest
    assert script_module.build_thesis_wrapper_plan is thesis_wrapper.build_thesis_wrapper_plan
    assert script_module.build_thesis_wrapper_plan_for_request is thesis_wrapper.build_thesis_wrapper_plan_for_request
    assert script_module.build_thesis_wrapper_commands_for_request is (
        thesis_wrapper.build_thesis_wrapper_commands_for_request
    )
    assert script_module.run_thesis_wrapper_cli is thesis_wrapper.run_thesis_wrapper_cli
    assert script_module.run_thesis_wrapper_commands is thesis_wrapper.run_thesis_wrapper_commands
    assert script_module.run_thesis_wrapper_plan is thesis_wrapper.run_thesis_wrapper_plan
    assert script_module.thesis_wrapper_repo_root is thesis_wrapper.thesis_wrapper_repo_root
    assert script_module.thesis_wrapper_commands_for_plan is thesis_wrapper.thesis_wrapper_commands_for_plan
    assert script_module.thesis_wrapper_inputs_from_args is thesis_wrapper.thesis_wrapper_inputs_from_args
    assert script_module.thesis_wrapper_request is thesis_wrapper.thesis_wrapper_request
    assert script_module.write_thesis_wrapper_summary is thesis_wrapper.write_thesis_wrapper_summary
    assert thesis_wrapper.main is thesis_wrapper_cli.main
    assert thesis_wrapper.build_thesis_wrapper_parser is thesis_wrapper_cli.build_thesis_wrapper_parser
    assert thesis_wrapper.run_thesis_wrapper_cli is thesis_wrapper_cli.run_thesis_wrapper_cli
    assert thesis_wrapper.thesis_wrapper_repo_root is thesis_wrapper_cli.thesis_wrapper_repo_root
    assert thesis_wrapper.ThesisWrapperPlan is thesis_wrapper_state.ThesisWrapperPlan
    assert thesis_wrapper.ThesisWrapperRequest is thesis_wrapper_state.ThesisWrapperRequest
    assert thesis_wrapper.ThesisWrapperResult is thesis_wrapper_state.ThesisWrapperResult
    assert thesis_wrapper.ThesisWrapperCommands is thesis_wrapper_plan_commands.ThesisWrapperCommands
    assert thesis_wrapper_plan.ThesisWrapperPlan is thesis_wrapper_state.ThesisWrapperPlan
    assert thesis_wrapper_plan.ThesisWrapperResult is thesis_wrapper_state.ThesisWrapperResult
    assert thesis_wrapper_plan.ThesisWrapperCommands is thesis_wrapper_plan_commands.ThesisWrapperCommands
    assert thesis_wrapper.build_thesis_wrapper_commands is thesis_wrapper_plan_commands.build_thesis_wrapper_commands
    assert thesis_wrapper.build_thesis_wrapper_commands_for_request is (
        thesis_wrapper_plan_commands.build_thesis_wrapper_commands_for_request
    )
    assert (
        thesis_wrapper_plan.build_thesis_wrapper_commands is thesis_wrapper_plan_commands.build_thesis_wrapper_commands
    )
    assert thesis_wrapper_plan.build_thesis_wrapper_commands_for_request is (
        thesis_wrapper_plan_commands.build_thesis_wrapper_commands_for_request
    )
    assert thesis_wrapper_plan.build_thesis_wrapper_plan_for_request is (
        thesis_wrapper.build_thesis_wrapper_plan_for_request
    )
    assert thesis_wrapper.thesis_wrapper_request is thesis_wrapper_request.thesis_wrapper_request
    assert thesis_wrapper_plan.thesis_wrapper_request is thesis_wrapper_request.thesis_wrapper_request
    assert thesis_wrapper.run_thesis_wrapper_commands is thesis_wrapper_plan_execution.run_thesis_wrapper_commands
    assert thesis_wrapper.thesis_wrapper_commands_for_plan is (
        thesis_wrapper_plan_execution.thesis_wrapper_commands_for_plan
    )
    assert thesis_wrapper_plan.run_thesis_wrapper_commands is thesis_wrapper_plan_execution.run_thesis_wrapper_commands
    assert thesis_wrapper_plan.thesis_wrapper_commands_for_plan is (
        thesis_wrapper_plan_execution.thesis_wrapper_commands_for_plan
    )
    assert thesis_wrapper.thesis_wrapper_summary_payload is thesis_wrapper_summary.thesis_wrapper_summary_payload
    assert thesis_wrapper.write_thesis_wrapper_summary is thesis_wrapper_summary.write_thesis_wrapper_summary
    assert thesis_wrapper_plan.thesis_wrapper_summary_payload is thesis_wrapper_summary.thesis_wrapper_summary_payload
    assert thesis_wrapper_plan.write_thesis_wrapper_summary is thesis_wrapper_summary.write_thesis_wrapper_summary


def test_thesis_wrapper_cli_parser_preserves_defaults_and_repeatable_args(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_cli import build_thesis_wrapper_parser, thesis_wrapper_repo_root

    parser = build_thesis_wrapper_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(tmp_path / "repo"),
            "--run-label",
            "demo_run",
            "--compare-run-dir",
            "runs/a",
            "--compare-run-dir",
            "runs/b",
            "--train-arg=--override",
            "--train-arg",
            "training.profile_timers=true",
        ]
    )

    assert args.preset == "standard"
    assert args.eval_preset == ""
    assert args.num_envs == 2
    assert args.unroll_length == 4
    assert args.max_updates == 1
    assert args.runtime_mode == "train_ordered"
    assert args.compare_run_dir == ["runs/a", "runs/b"]
    assert args.train_arg == ["--override", "training.profile_timers=true"]
    assert thesis_wrapper_repo_root(args) == (tmp_path / "repo").resolve()


def test_thesis_wrapper_cli_lists_presets_without_requiring_run_label(tmp_path: Path, capsys) -> None:
    from weiss_rl.workflows.thesis_wrapper_cli import build_thesis_wrapper_parser, run_thesis_wrapper_cli

    repo_root = tmp_path / "repo"
    parser = build_thesis_wrapper_parser()
    args = parser.parse_args(["--repo-root", str(repo_root), "--list-presets"])

    status = run_thesis_wrapper_cli(args=args, parser=parser, repo_root=repo_root, python_exe="python.exe")

    assert status == 0
    output = capsys.readouterr().out
    assert f"standard: {(repo_root / 'configs/presets/structured_acceptance_standard.yaml').as_posix()}" in output
    assert "standard-auto-gpu:" in output
    assert "ablate-no-tactical-bias:" in output


def test_thesis_wrapper_input_boundary_normalizes_argparse_namespace(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_inputs import ThesisWrapperInputs, thesis_wrapper_inputs_from_args

    args = SimpleNamespace(
        run_label="demo_run",
        stack_config=Path("configs/train.yaml"),
        eval_stack_config=None,
        preset="standard",
        eval_preset="",
        num_envs="8",
        unroll_length="16",
        max_updates="32",
        runtime_mode="train_ordered",
        profile="fast",
        device="cuda",
        seed=7,
        resume_run_dir=tmp_path / "runs" / "resume_source",
        resume_from="latest",
        b1_baseline_run_dir=tmp_path / "runs" / "b1",
        compare_run_dir=[tmp_path / "runs" / "baseline_a", "runs/baseline_b"],
        compare_launch_group_summary=tmp_path / "launch_summary.json",
        compare_out_dir=tmp_path / "compare_out",
        train_arg=["--override", "training.profile_timers=true"],
        eval_arg=["--paired-seed-limit", 4],
        compare_arg=["--format", "md"],
        skip_eval=False,
        skip_compare=True,
        dry_run=True,
    )

    inputs = thesis_wrapper_inputs_from_args(args)

    assert isinstance(inputs, ThesisWrapperInputs)
    assert inputs.num_envs == 8
    assert inputs.unroll_length == 16
    assert inputs.max_updates == 32
    assert inputs.compare_run_dirs == (str(tmp_path / "runs" / "baseline_a"), "runs/baseline_b")
    assert inputs.train_args == ("--override", "training.profile_timers=true")
    assert inputs.eval_args == ("--paired-seed-limit", "4")
    assert inputs.compare_args == ("--format", "md")
    assert inputs.skip_compare is True
    assert inputs.dry_run is True


def test_thesis_wrapper_command_selection_preserves_optional_steps(tmp_path: Path) -> None:
    from dataclasses import replace

    from weiss_rl.workflows.thesis_wrapper_inputs import ThesisWrapperInputs
    from weiss_rl.workflows.thesis_wrapper_plan_commands import (
        ThesisWrapperCommands,
        build_thesis_wrapper_commands,
    )

    inputs = ThesisWrapperInputs(
        run_label="demo_run",
        stack_config=None,
        eval_stack_config=None,
        preset="standard",
        eval_preset="",
        num_envs=8,
        unroll_length=16,
        max_updates=32,
        runtime_mode="train_ordered",
        profile="fast",
        device="cuda",
        seed=7,
        resume_run_dir=tmp_path / "repo" / "runs" / "resume_source",
        resume_from="latest",
        b1_baseline_run_dir=tmp_path / "repo" / "runs" / "b1_anchor",
        compare_run_dirs=(str(tmp_path / "repo" / "runs" / "baseline_a"),),
        compare_launch_group_summary=tmp_path / "repo" / "runs" / "launch_summary.json",
        compare_out_dir=tmp_path / "repo" / "runs" / "compare_out",
        train_args=("--override", "training.profile_timers=true"),
        eval_args=("--paired-seed-limit", "4"),
        compare_args=("--format", "md"),
        skip_eval=False,
        skip_compare=False,
        dry_run=True,
    )

    commands = build_thesis_wrapper_commands(
        inputs=inputs,
        python_exe="python.exe",
        run_dir=tmp_path / "repo" / "runs" / "demo_run",
        stack_config=tmp_path / "repo" / "configs" / "train.yaml",
        eval_stack_config=tmp_path / "repo" / "configs" / "eval.yaml",
    )

    assert isinstance(commands, ThesisWrapperCommands)
    assert commands.train_command[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert commands.train_command[-2:] == ["--override", "training.profile_timers=true"]
    assert commands.eval_command is not None
    assert commands.eval_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"]
    assert commands.eval_command[-2:] == ["--paired-seed-limit", "4"]
    assert commands.compare_command is not None
    assert commands.compare_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.compare_runs_entrypoint"]
    assert commands.compare_command[-2:] == ["--format", "md"]

    skip_inputs = replace(inputs, skip_eval=True, skip_compare=True)
    skip_commands = build_thesis_wrapper_commands(
        inputs=skip_inputs,
        python_exe="python.exe",
        run_dir=tmp_path / "repo" / "runs" / "demo_run",
        stack_config=tmp_path / "repo" / "configs" / "train.yaml",
        eval_stack_config=tmp_path / "repo" / "configs" / "eval.yaml",
    )

    assert skip_commands.train_command == commands.train_command
    assert skip_commands.eval_command is None
    assert skip_commands.compare_command is None


def test_thesis_wrapper_plan_execution_runs_selected_commands_in_order(tmp_path: Path) -> None:
    from dataclasses import replace

    from weiss_rl.workflows.thesis_wrapper_plan_execution import (
        run_thesis_wrapper_commands,
        thesis_wrapper_commands_for_plan,
    )
    from weiss_rl.workflows.thesis_wrapper_state import ThesisWrapperPlan, ThesisWrapperResult

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
        compare_command=["python.exe", "-m", "weiss_rl.workflows.compare_runs_entrypoint"],
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
    from weiss_rl.workflows.thesis_wrapper_plan_execution import run_thesis_wrapper_commands
    from weiss_rl.workflows.thesis_wrapper_state import ThesisWrapperPlan

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
        compare_command=["python.exe", "-m", "weiss_rl.workflows.compare_runs_entrypoint"],
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


def test_thesis_wrapper_plan_builder_preserves_default_eval_and_optional_commands(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_plan import (
        ThesisWrapperRequest,
        build_thesis_wrapper_commands_for_request,
        build_thesis_wrapper_plan,
        build_thesis_wrapper_plan_for_request,
        thesis_wrapper_request,
        thesis_wrapper_summary_payload,
    )

    repo_root = tmp_path / "repo"
    b1_run = repo_root / "runs" / "b1_anchor"
    args = SimpleNamespace(
        run_label="main_run",
        stack_config=None,
        eval_stack_config=None,
        preset="standard",
        eval_preset="",
        num_envs=8,
        unroll_length=16,
        max_updates=32,
        runtime_mode="train_ordered",
        profile="fast",
        device="cuda",
        seed=7,
        resume_run_dir=None,
        resume_from="",
        b1_baseline_run_dir=b1_run,
        compare_run_dir=[repo_root / "runs" / "baseline"],
        compare_launch_group_summary=repo_root / "runs" / "launch_summary.json",
        compare_out_dir=repo_root / "runs" / "compare_out",
        train_arg=["--override", "training.profile_timers=true"],
        eval_arg=["--paired-seed-limit", "4"],
        compare_arg=["--format", "md"],
        skip_eval=False,
        skip_compare=False,
        dry_run=True,
    )

    request = thesis_wrapper_request(args=args, repo_root=repo_root, python_exe="python.exe")
    request_commands = build_thesis_wrapper_commands_for_request(request)
    plan = build_thesis_wrapper_plan_for_request(request)
    legacy_plan = build_thesis_wrapper_plan(args=args, repo_root=repo_root, python_exe="python.exe")
    payload = thesis_wrapper_summary_payload(
        SimpleNamespace(plan=plan, steps=[{"status": "planned"}], failed=False, status="planned")
    )

    assert isinstance(request, ThesisWrapperRequest)
    assert request.run_label == "main_run"
    assert request.dry_run is True
    assert request.run_dir == repo_root / "runs" / "main_run"
    assert request_commands.train_command == plan.train_command
    assert plan == legacy_plan
    assert plan.stack_config == (repo_root / "configs" / "presets" / "structured_acceptance_standard.yaml").resolve()
    assert plan.eval_preset == "standard-thesis-eval"
    assert (
        plan.eval_stack_config
        == (repo_root / "configs" / "presets" / "structured_acceptance_standard_thesis_eval.yaml").resolve()
    )
    assert plan.train_command[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert plan.train_command[-2:] == ["--override", "training.profile_timers=true"]
    assert plan.eval_command is not None
    assert plan.eval_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"]
    assert plan.eval_command[-2:] == ["--paired-seed-limit", "4"]
    assert plan.compare_command is not None
    assert plan.compare_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.compare_runs_entrypoint"]
    assert payload["status"] == "planned"
    assert payload["b1_baseline_run_dir"] == b1_run.resolve().as_posix()


def test_thesis_wrapper_plan_runner_records_failed_step_and_summary(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from weiss_rl.workflows import thesis_wrapper_plan
    from weiss_rl.workflows.thesis_wrapper_plan import (
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


def test_thesis_wrapper_command_builders_preserve_train_eval_compare_shapes(tmp_path: Path) -> None:
    from weiss_rl.workflows import thesis_wrapper_commands
    from weiss_rl.workflows.entrypoint_command_builders import (
        build_eval_entrypoint_command,
        build_train_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_command_builders import (
        build_eval_entrypoint_command as wrapper_eval_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_command_builders import (
        build_thesis_compare_command as package_build_compare,
    )
    from weiss_rl.workflows.thesis_wrapper_command_builders import (
        build_train_entrypoint_command as wrapper_train_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_commands import (
        build_thesis_compare_command,
        build_thesis_eval_command,
        build_thesis_train_command,
    )

    assert thesis_wrapper_commands.build_thesis_compare_command is package_build_compare
    assert wrapper_train_entrypoint_command is build_train_entrypoint_command
    assert wrapper_eval_entrypoint_command is build_eval_entrypoint_command

    run_dir = tmp_path / "repo" / "runs" / "demo_run"
    b1_run = tmp_path / "repo" / "runs" / "b1_anchor"
    train = build_thesis_train_command(
        python_exe="python.exe",
        stack_config=tmp_path / "repo" / "configs" / "stack.yaml",
        run_label="demo_run",
        num_envs=8,
        unroll_length=16,
        max_updates=32,
        runtime_mode="train_ordered",
        profile="fast",
        device="cuda",
        seed=7,
        resume_run_dir=tmp_path / "repo" / "runs" / "resume_source",
        resume_from="latest",
        b1_baseline_run_dir=b1_run,
        train_args=("--override", "training.profile_timers=true"),
    )
    eval_command = build_thesis_eval_command(
        python_exe="python.exe",
        eval_stack_config=tmp_path / "repo" / "configs" / "eval.yaml",
        run_dir=run_dir,
        b1_baseline_run_dir=b1_run,
        eval_args=("--paired-seed-limit", "2"),
    )
    compare = build_thesis_compare_command(
        python_exe="python.exe",
        run_dir=run_dir,
        compare_run_dirs=(str(tmp_path / "repo" / "runs" / "baseline_a"),),
        compare_launch_group_summary=tmp_path / "repo" / "runs" / "launch_summary.json",
        compare_out_dir=tmp_path / "repo" / "runs" / "compare_out",
        compare_args=("--format", "md"),
    )

    assert train[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert "--stack-config" in train
    assert str(tmp_path / "repo" / "configs" / "stack.yaml") in train
    assert "--run-label" in train
    assert "demo_run" in train
    assert "--seed" in train
    assert "7" in train
    assert "--resume-run-dir" in train
    assert str(tmp_path / "repo" / "runs" / "resume_source") in train
    assert train[-2:] == ["--override", "training.profile_timers=true"]

    assert eval_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"]
    assert "--run-dir" in eval_command
    assert str(run_dir) in eval_command
    assert "--b1-baseline-run-dir" in eval_command
    assert str(b1_run) in eval_command
    assert eval_command[-2:] == ["--paired-seed-limit", "2"]

    assert compare[:3] == ["python.exe", "-m", "weiss_rl.workflows.compare_runs_entrypoint"]
    assert compare.count("--run-dir") == 2
    assert str(tmp_path / "repo" / "runs" / "baseline_a") in compare
    assert "--launch-group-summary" in compare
    assert str(tmp_path / "repo" / "runs" / "launch_summary.json") in compare
    assert "--out-dir" in compare
    assert str(tmp_path / "repo" / "runs" / "compare_out") in compare
    assert compare[-2:] == ["--format", "md"]


def test_thesis_run_wrapper_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "demo_run",
            "--dry-run",
            "--compare-run-dir",
            str(repo_root / "runs" / "baseline_a"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "demo_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][1]["command"][1:3] == ["-m", "weiss_rl.workflows.eval_entrypoint"]
    assert payload["steps"][2]["command"][1:3] == ["-m", "weiss_rl.workflows.compare_runs_entrypoint"]


def test_thesis_run_package_module_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.workflows.thesis_wrapper",
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "module_demo_run",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "module_demo_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "thesis_run_wrapper_v1"
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][1]["command"][1:3] == ["-m", "weiss_rl.workflows.eval_entrypoint"]


def test_thesis_run_wrapper_defaults_to_standard_preset_when_stack_config_is_omitted(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--run-label",
            "default_preset_run",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "default_preset_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "standard"
    assert payload["stack_config"].endswith("configs/presets/structured_acceptance_standard.yaml")
    assert payload["eval_preset"] == "standard-thesis-eval"
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_standard_thesis_eval.yaml")


def test_thesis_run_wrapper_reuses_custom_stack_config_for_eval_when_no_eval_override_is_supplied(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "custom_eval_match",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "custom_eval_match.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_preset"] == ""


def test_thesis_run_wrapper_resolves_relative_config_paths_against_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "train_stack.yaml"
    eval_stack_config = repo_root / "configs" / "eval_stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    eval_stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    invocation_cwd = tmp_path / "outside_repo"
    invocation_cwd.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            "configs/train_stack.yaml",
            "--eval-stack-config",
            "configs/eval_stack.yaml",
            "--run-label",
            "relative_paths",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=invocation_cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "relative_paths.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == eval_stack_config.resolve().as_posix()


def test_thesis_run_wrapper_defaults_to_multideck_eval_surface_for_multideck_preset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets" / "ablations").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--preset",
            "standard-multideck",
            "--run-label",
            "multideck_default_eval",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "multideck_default_eval.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "standard-multideck"
    assert payload["eval_preset"] == "standard-multideck"
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_standard_multideck.yaml")


def test_thesis_run_wrapper_lists_named_presets_without_run_label(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--list-presets",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "standard:" in result.stdout
    assert "standard-auto-gpu:" in result.stdout
    assert "ablate-teacher-fade:" in result.stdout
    assert "ablate-no-tactical-bias:" in result.stdout


def test_thesis_run_wrapper_passes_b1_baseline_run_dir_to_train_and_eval(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    baseline_run_dir = repo_root / "runs" / "b1_anchor_seed1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "baseline_passthrough",
            "--b1-baseline-run-dir",
            str(baseline_run_dir),
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "baseline_passthrough.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["b1_baseline_run_dir"] == baseline_run_dir.resolve().as_posix()
    assert "--b1-baseline-run-dir" in payload["steps"][0]["command"]
    assert str(baseline_run_dir) in payload["steps"][0]["command"]
    assert "--b1-baseline-run-dir" in payload["steps"][1]["command"]
    assert str(baseline_run_dir) in payload["steps"][1]["command"]
