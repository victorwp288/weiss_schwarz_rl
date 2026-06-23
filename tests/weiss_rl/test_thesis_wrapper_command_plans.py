from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_thesis_wrapper_command_selection_preserves_optional_steps(tmp_path: Path) -> None:
    from dataclasses import replace

    from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs
    from weiss_rl.workflows.thesis_wrapper_support.plan_commands import (
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
    assert commands.compare_command[:3] == [
        "python.exe",
        "-m",
        "weiss_rl.workflows.compare_runs.compare_runs_entrypoint",
    ]
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


def test_thesis_wrapper_plan_builder_preserves_default_eval_and_optional_commands(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_support.plan import (
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
    assert plan.compare_command[:3] == ["python.exe", "-m", "weiss_rl.workflows.compare_runs.compare_runs_entrypoint"]
    assert payload["status"] == "planned"
    assert payload["b1_baseline_run_dir"] == b1_run.resolve().as_posix()


def test_thesis_wrapper_command_builders_preserve_train_eval_compare_shapes(tmp_path: Path) -> None:
    import weiss_rl.workflows.thesis_wrapper_support.commands as thesis_wrapper_commands
    from weiss_rl.workflows.entrypoint_command_builders import (
        build_eval_entrypoint_command,
        build_train_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_support.command_builders import (
        build_eval_entrypoint_command as wrapper_eval_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_support.command_builders import (
        build_thesis_compare_command as package_build_compare,
    )
    from weiss_rl.workflows.thesis_wrapper_support.command_builders import (
        build_train_entrypoint_command as wrapper_train_entrypoint_command,
    )
    from weiss_rl.workflows.thesis_wrapper_support.commands import (
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

    assert compare[:3] == ["python.exe", "-m", "weiss_rl.workflows.compare_runs.compare_runs_entrypoint"]
    assert compare.count("--run-dir") == 2
    assert str(tmp_path / "repo" / "runs" / "baseline_a") in compare
    assert "--launch-group-summary" in compare
    assert str(tmp_path / "repo" / "runs" / "launch_summary.json") in compare
    assert "--out-dir" in compare
    assert str(tmp_path / "repo" / "runs" / "compare_out") in compare
    assert compare[-2:] == ["--format", "md"]
