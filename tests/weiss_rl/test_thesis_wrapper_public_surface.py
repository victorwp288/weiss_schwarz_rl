from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .thesis_run_wrapper_test_support import REPO_ROOT


def test_thesis_wrapper_package_module_reexports_workflow_facade() -> None:
    import weiss_rl.workflows.thesis_wrapper as thesis_wrapper
    import weiss_rl.workflows.thesis_wrapper_support.cli as thesis_wrapper_cli
    import weiss_rl.workflows.thesis_wrapper_support.plan as thesis_wrapper_plan
    import weiss_rl.workflows.thesis_wrapper_support.plan_commands as thesis_wrapper_plan_commands
    import weiss_rl.workflows.thesis_wrapper_support.plan_execution as thesis_wrapper_plan_execution
    import weiss_rl.workflows.thesis_wrapper_support.request as thesis_wrapper_request
    import weiss_rl.workflows.thesis_wrapper_support.state as thesis_wrapper_state
    import weiss_rl.workflows.thesis_wrapper_support.summary as thesis_wrapper_summary
    from weiss_rl.workflows.thesis_wrapper_support import command_builders, commands, inputs

    assert thesis_wrapper.main is thesis_wrapper_cli.main
    assert thesis_wrapper._PRESET_PATHS is commands._PRESET_PATHS
    assert thesis_wrapper._resolve_stack_config is commands._resolve_stack_config
    assert thesis_wrapper._resolve_eval_stack_config is commands._resolve_eval_stack_config
    assert thesis_wrapper.build_thesis_train_command is command_builders.build_thesis_train_command
    assert thesis_wrapper.build_thesis_eval_command is command_builders.build_thesis_eval_command
    assert thesis_wrapper.build_thesis_compare_command is command_builders.build_thesis_compare_command
    assert thesis_wrapper.build_thesis_wrapper_parser is thesis_wrapper_cli.build_thesis_wrapper_parser
    assert thesis_wrapper.run_thesis_wrapper_cli is thesis_wrapper_cli.run_thesis_wrapper_cli
    assert thesis_wrapper.thesis_wrapper_repo_root is thesis_wrapper_cli.thesis_wrapper_repo_root
    assert thesis_wrapper.ThesisWrapperInputs is inputs.ThesisWrapperInputs
    assert thesis_wrapper.thesis_wrapper_inputs_from_args is inputs.thesis_wrapper_inputs_from_args
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
    from weiss_rl.workflows.thesis_wrapper_support.cli import build_thesis_wrapper_parser, thesis_wrapper_repo_root

    parser = build_thesis_wrapper_parser()
    default_args = parser.parse_args(["--run-label", "demo_run"])
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
    assert thesis_wrapper_repo_root(default_args) == REPO_ROOT
    assert thesis_wrapper_repo_root(args) == (tmp_path / "repo").resolve()


def test_thesis_wrapper_cli_lists_presets_without_requiring_run_label(tmp_path: Path, capsys) -> None:
    from weiss_rl.workflows.thesis_wrapper_support.cli import build_thesis_wrapper_parser, run_thesis_wrapper_cli

    repo_root = tmp_path / "repo"
    parser = build_thesis_wrapper_parser()
    args = parser.parse_args(["--repo-root", str(repo_root), "--list-presets"])

    status = run_thesis_wrapper_cli(args=args, parser=parser, repo_root=repo_root, python_exe="python.exe")

    assert status == 0
    output = capsys.readouterr().out
    assert f"standard: {(repo_root / 'configs/presets/structured_acceptance_standard.yaml').as_posix()}" in output
    assert "standard-auto-gpu:" in output
    assert "ablate-no-gru:" in output


def test_thesis_wrapper_input_boundary_normalizes_argparse_namespace(tmp_path: Path) -> None:
    from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs, thesis_wrapper_inputs_from_args

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
