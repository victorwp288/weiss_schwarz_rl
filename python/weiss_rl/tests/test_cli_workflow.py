from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_cli_b1_source_run(run_dir: Path, *, policy_id: str = "selected_candidate", update: int = 15) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": policy_id,
                        "update": update,
                        "weights_sha256": "a" * 64,
                        "path": f"training/snapshots/{policy_id}/weights.pt",
                    }
                ],
                "champion_snapshots": [],
                "pinned_snapshots": [policy_id],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_package_cli_facade_keeps_legacy_helper_imports() -> None:
    import weiss_rl.cli as cli
    from weiss_rl.workflows import (
        commands,
        controller_commands,
        controller_dispatch,
        controller_execution,
        controller_guard_commands,
        controller_guard_plan,
        controller_guarded_league_commands,
        controller_guarded_league_plan,
        controller_guided_commands,
        controller_guided_plan,
        controller_parser,
        controller_parser_arguments,
        controller_plan,
        controller_plan_state,
        entrypoint_command_builders,
        evaluation_audit_commands,
        evaluation_audit_plan,
        evaluation_command_config,
        evaluation_commands,
        evaluation_dispatch,
        evaluation_eval_commands,
        evaluation_eval_plan,
        evaluation_execution,
        evaluation_figure_commands,
        evaluation_figure_plan,
        evaluation_parser,
        evaluation_parser_arguments,
        evaluation_plan,
        evaluation_plan_state,
        parsers,
        planning,
        public_api,
        runner,
        training_baseline_plan,
        training_command_builders,
        training_commands,
        training_dispatch,
        training_execution,
        training_main_plan,
        training_parser,
        training_parser_arguments,
        training_plan,
        training_plan_state,
        training_profiles,
        training_snapshot_resolution,
        workflow_dispatch,
    )
    from weiss_rl.workflows import (
        controllers as controllers_facade,
    )
    from weiss_rl.workflows import (
        evaluation as evaluation_facade,
    )
    from weiss_rl.workflows import (
        training as training_facade,
    )

    assert cli.main is runner.main
    assert cli.ControllerWorkflowPlan is commands.ControllerWorkflowPlan
    assert cli.ControllerWorkflowRequest is commands.ControllerWorkflowRequest
    assert cli.build_controller_workflow_plan is commands.build_controller_workflow_plan
    assert cli.controller_workflow_request is commands.controller_workflow_request
    assert cli.EvaluationWorkflowPlan is commands.EvaluationWorkflowPlan
    assert cli.EvaluationWorkflowRequest is commands.EvaluationWorkflowRequest
    assert cli.build_evaluation_workflow_plan is commands.build_evaluation_workflow_plan
    assert cli.evaluation_workflow_request is commands.evaluation_workflow_request
    assert cli.TRAIN_PROFILES is commands.TRAIN_PROFILES
    assert cli.TrainProfile is commands.TrainProfile
    assert cli.TrainingWorkflowPlan is commands.TrainingWorkflowPlan
    assert cli.TrainingWorkflowRequest is commands.TrainingWorkflowRequest
    assert cli.WorkflowDispatchRequest is commands.WorkflowDispatchRequest
    assert cli.build_training_workflow_plan is commands.build_training_workflow_plan
    assert cli.dispatch_workflow_command is commands.dispatch_workflow_command
    assert cli.training_workflow_request is commands.training_workflow_request
    assert cli.workflow_dispatch_request is commands.workflow_dispatch_request
    assert cli._add_common is runner._add_common
    assert cli._train_command is commands._train_command
    assert cli._eval_command is commands._eval_command
    assert cli._parse_args is runner._parse_args
    assert cli._repo_root(None) == REPO_ROOT
    assert training_facade.add_training_parsers is training_parser.add_training_parsers
    assert training_facade.dispatch_training_command is training_dispatch.dispatch_training_command
    assert training_facade.dispatch_training_request is training_dispatch.dispatch_training_request
    assert training_facade.TrainingWorkflowRequest is training_plan_state.TrainingWorkflowRequest
    assert training_facade._run_training_workflow_plan is training_execution._run_training_workflow_plan
    assert evaluation_facade.add_evaluation_parsers is evaluation_parser.add_evaluation_parsers
    assert evaluation_facade.dispatch_evaluation_command is evaluation_dispatch.dispatch_evaluation_command
    assert evaluation_facade.dispatch_evaluation_request is evaluation_dispatch.dispatch_evaluation_request
    assert evaluation_facade.EvaluationWorkflowRequest is evaluation_plan_state.EvaluationWorkflowRequest
    assert evaluation_facade._run_evaluation_workflow_plan is evaluation_execution._run_evaluation_workflow_plan
    assert controllers_facade.add_controller_parsers is controller_parser.add_controller_parsers
    assert controllers_facade.dispatch_controller_command is controller_dispatch.dispatch_controller_command
    assert controllers_facade.dispatch_controller_request is controller_dispatch.dispatch_controller_request
    assert controllers_facade.ControllerWorkflowRequest is controller_plan_state.ControllerWorkflowRequest
    assert controllers_facade._run_controller_workflow_plan is controller_execution._run_controller_workflow_plan
    assert parsers.add_evaluation_parsers is evaluation_parser.add_evaluation_parsers
    assert parsers.add_controller_parsers is controller_parser.add_controller_parsers
    assert runner.dispatch_training_command is training_dispatch.dispatch_training_command
    assert runner.dispatch_evaluation_command is evaluation_dispatch.dispatch_evaluation_command
    assert runner.dispatch_controller_command is controller_dispatch.dispatch_controller_command
    assert runner.dispatch_workflow_command is workflow_dispatch.dispatch_workflow_command
    assert set(commands.__all__) == set(public_api.PUBLIC_WORKFLOW_EXPORTS)
    assert set(public_api.PUBLIC_WORKFLOW_EXPORTS).issubset(set(runner.__all__))
    assert set(public_api.PUBLIC_WORKFLOW_EXPORTS).issubset(set(cli.__all__))
    assert commands._train_command is training_commands._train_command
    assert commands._train_command is training_command_builders._train_command
    assert training_command_builders.build_train_entrypoint_command is (
        entrypoint_command_builders.build_train_entrypoint_command
    )
    assert commands.TrainProfile is training_profiles.TrainProfile
    assert commands.TRAIN_PROFILES is training_profiles.TRAIN_PROFILES
    assert commands.TrainingWorkflowPlan is training_plan_state.TrainingWorkflowPlan
    assert commands.TrainingWorkflowRequest is training_plan_state.TrainingWorkflowRequest
    assert commands.training_workflow_request is training_plan_state.training_workflow_request
    assert training_plan.TrainingWorkflowPlan is training_plan_state.TrainingWorkflowPlan
    assert training_plan.TrainingWorkflowRequest is training_plan_state.TrainingWorkflowRequest
    assert training_plan.training_workflow_request is training_plan_state.training_workflow_request
    assert commands.build_training_workflow_plan is training_plan.build_training_workflow_plan
    assert training_plan.build_b1_training_workflow_plan is training_baseline_plan.build_b1_training_workflow_plan
    assert training_plan.build_b1_training_workflow_plan_for_request is (
        training_baseline_plan.build_b1_training_workflow_plan_for_request
    )
    assert training_plan.build_b1_guided_seed_training_workflow_plan is (
        training_baseline_plan.build_b1_guided_seed_training_workflow_plan
    )
    assert training_plan.build_b1_guided_seed_training_workflow_plan_for_request is (
        training_baseline_plan.build_b1_guided_seed_training_workflow_plan_for_request
    )
    assert training_plan.build_main_training_workflow_plan is training_main_plan.build_main_training_workflow_plan
    assert training_plan.build_main_training_workflow_plan_for_request is (
        training_main_plan.build_main_training_workflow_plan_for_request
    )
    assert training_plan.build_main_guided_bootstrap_training_workflow_plan is (
        training_main_plan.build_main_guided_bootstrap_training_workflow_plan
    )
    assert training_plan.build_main_guided_bootstrap_training_workflow_plan_for_request is (
        training_main_plan.build_main_guided_bootstrap_training_workflow_plan_for_request
    )
    assert training_commands.build_b1_training_workflow_plan is training_baseline_plan.build_b1_training_workflow_plan
    assert training_commands.build_main_training_workflow_plan is training_main_plan.build_main_training_workflow_plan
    assert commands._resolve_snapshot_checkpoint_path is training_snapshot_resolution._resolve_snapshot_checkpoint_path
    assert training_parser.add_train_b1_parser is training_parser_arguments.add_train_b1_parser
    assert training_parser.add_train_b1_guided_seed_parser is training_parser_arguments.add_train_b1_guided_seed_parser
    assert training_parser.add_train_main_parser is training_parser_arguments.add_train_main_parser
    assert training_parser.add_train_main_guided_bootstrap_parser is (
        training_parser_arguments.add_train_main_guided_bootstrap_parser
    )
    assert commands._eval_command is evaluation_commands._eval_command
    assert commands._eval_command is evaluation_eval_commands._eval_command
    assert evaluation_commands.EVAL_STACK_CONFIG is evaluation_command_config.EVAL_STACK_CONFIG
    assert evaluation_commands.build_eval_entrypoint_command is (
        entrypoint_command_builders.build_eval_entrypoint_command
    )
    assert commands._figures_command is evaluation_commands._figures_command
    assert commands._figures_command is evaluation_figure_commands._figures_command
    assert commands._b2_audit_command is evaluation_commands._b2_audit_command
    assert commands._b2_audit_command is evaluation_audit_commands._b2_audit_command
    assert commands.EvaluationWorkflowPlan is evaluation_plan_state.EvaluationWorkflowPlan
    assert commands.EvaluationWorkflowRequest is evaluation_plan_state.EvaluationWorkflowRequest
    assert commands.evaluation_workflow_request is evaluation_plan_state.evaluation_workflow_request
    assert evaluation_plan.EvaluationWorkflowPlan is evaluation_plan_state.EvaluationWorkflowPlan
    assert evaluation_plan.EvaluationWorkflowRequest is evaluation_plan_state.EvaluationWorkflowRequest
    assert evaluation_plan.evaluation_workflow_request is evaluation_plan_state.evaluation_workflow_request
    assert commands.build_evaluation_workflow_plan is evaluation_plan.build_evaluation_workflow_plan
    assert evaluation_plan.build_eval_workflow_plan is evaluation_eval_plan.build_eval_workflow_plan
    assert (
        evaluation_plan.build_eval_workflow_plan_for_request
        is evaluation_eval_plan.build_eval_workflow_plan_for_request
    )
    assert evaluation_plan.build_figures_workflow_plan is evaluation_figure_plan.build_figures_workflow_plan
    assert evaluation_plan.build_figures_workflow_plan_for_request is (
        evaluation_figure_plan.build_figures_workflow_plan_for_request
    )
    assert evaluation_plan.build_b2_audit_workflow_plan is evaluation_audit_plan.build_b2_audit_workflow_plan
    assert evaluation_plan.build_b2_audit_workflow_plan_for_request is (
        evaluation_audit_plan.build_b2_audit_workflow_plan_for_request
    )
    assert evaluation_parser.add_smoke_eval_parser is evaluation_parser_arguments.add_smoke_eval_parser
    assert evaluation_parser.add_eval_final_parser is evaluation_parser_arguments.add_eval_final_parser
    assert evaluation_parser.add_figures_parser is evaluation_parser_arguments.add_figures_parser
    assert evaluation_parser.add_b2_audit_parser is evaluation_parser_arguments.add_b2_audit_parser
    assert commands._guard_run_command is controller_commands._guard_run_command
    assert commands._guard_run_command is controller_guard_commands._guard_run_command
    assert commands._guided_bootstrap_loop_command is controller_commands._guided_bootstrap_loop_command
    assert commands._guided_bootstrap_loop_command is controller_guided_commands._guided_bootstrap_loop_command
    assert commands._guarded_league_bootstrap_command is controller_commands._guarded_league_bootstrap_command
    assert commands._guarded_league_bootstrap_command is (
        controller_guarded_league_commands._guarded_league_bootstrap_command
    )
    assert commands.ControllerWorkflowPlan is controller_plan_state.ControllerWorkflowPlan
    assert commands.ControllerWorkflowRequest is controller_plan_state.ControllerWorkflowRequest
    assert commands.controller_workflow_request is controller_plan_state.controller_workflow_request
    assert controller_plan.ControllerWorkflowPlan is controller_plan_state.ControllerWorkflowPlan
    assert controller_plan.ControllerWorkflowRequest is controller_plan_state.ControllerWorkflowRequest
    assert controller_plan.controller_workflow_request is controller_plan_state.controller_workflow_request
    assert commands.build_controller_workflow_plan is controller_plan.build_controller_workflow_plan
    assert controller_plan.build_guard_run_workflow_plan is controller_guard_plan.build_guard_run_workflow_plan
    assert controller_plan.build_guard_run_workflow_plan_for_request is (
        controller_guard_plan.build_guard_run_workflow_plan_for_request
    )
    assert controller_plan.build_guided_bootstrap_loop_workflow_plan is (
        controller_guided_plan.build_guided_bootstrap_loop_workflow_plan
    )
    assert controller_plan.build_guided_bootstrap_loop_workflow_plan_for_request is (
        controller_guided_plan.build_guided_bootstrap_loop_workflow_plan_for_request
    )
    assert controller_plan.build_guarded_league_bootstrap_workflow_plan is (
        controller_guarded_league_plan.build_guarded_league_bootstrap_workflow_plan
    )
    assert controller_plan.build_guarded_league_bootstrap_workflow_plan_for_request is (
        controller_guarded_league_plan.build_guarded_league_bootstrap_workflow_plan_for_request
    )
    assert controller_plan._validate_guarded_league_args is controller_guarded_league_plan._validate_guarded_league_args
    assert controller_parser.add_guard_run_parser is controller_parser_arguments.add_guard_run_parser
    assert controller_parser.add_guided_bootstrap_loop_parser is (
        controller_parser_arguments.add_guided_bootstrap_loop_parser
    )
    assert controller_parser.add_guarded_league_bootstrap_parser is (
        controller_parser_arguments.add_guarded_league_bootstrap_parser
    )
    assert public_api.build_training_workflow_plan is training_plan.build_training_workflow_plan
    assert public_api.build_evaluation_workflow_plan is evaluation_plan.build_evaluation_workflow_plan
    assert public_api.build_controller_workflow_plan is controller_plan.build_controller_workflow_plan
    assert public_api.evaluation_workflow_request is evaluation_plan.evaluation_workflow_request
    assert public_api.controller_workflow_request is controller_plan.controller_workflow_request
    assert public_api.WorkflowDispatchRequest is workflow_dispatch.WorkflowDispatchRequest
    assert public_api.dispatch_workflow_command is workflow_dispatch.dispatch_workflow_command
    assert public_api.dispatch_workflow_request is workflow_dispatch.dispatch_workflow_request
    assert public_api.workflow_dispatch_request is workflow_dispatch.workflow_dispatch_request
    assert runner.build_evaluation_workflow_plan is public_api.build_evaluation_workflow_plan
    assert cli.build_evaluation_workflow_plan is public_api.build_evaluation_workflow_plan
    assert commands._run_or_plan is planning._run_or_plan


def test_workflow_dispatch_request_routes_in_family_order(tmp_path: Path, monkeypatch) -> None:
    from weiss_rl.workflows import workflow_dispatch
    from weiss_rl.workflows.controller_plan_state import ControllerWorkflowRequest
    from weiss_rl.workflows.evaluation_plan_state import EvaluationWorkflowRequest
    from weiss_rl.workflows.training_plan_state import TrainingWorkflowRequest

    args = SimpleNamespace(command="figures", dry_run=True)
    calls: list[tuple[str, str, Path, str]] = []

    def dispatch_training(request: TrainingWorkflowRequest) -> bool:
        calls.append(("training", request.command, request.repo_root, request.python_exe))
        return False

    def dispatch_evaluation(request: EvaluationWorkflowRequest) -> bool:
        calls.append(("evaluation", request.command, request.repo_root, request.python_exe))
        return True

    def dispatch_controller(request: ControllerWorkflowRequest) -> bool:
        calls.append(("controller", request.command, request.repo_root, request.python_exe))
        return True

    monkeypatch.setattr(workflow_dispatch, "dispatch_training_request", dispatch_training)
    monkeypatch.setattr(workflow_dispatch, "dispatch_evaluation_request", dispatch_evaluation)
    monkeypatch.setattr(workflow_dispatch, "dispatch_controller_request", dispatch_controller)

    request = workflow_dispatch.workflow_dispatch_request(args=args, repo_root=tmp_path, python_exe="python.exe")
    handled = workflow_dispatch.dispatch_workflow_request(request)

    assert request.command == "figures"
    assert request.dry_run is True
    assert request.evaluation_request().command == "figures"
    assert handled is True
    assert calls == [
        ("training", "figures", tmp_path, "python.exe"),
        ("evaluation", "figures", tmp_path, "python.exe"),
    ]


def test_runner_main_dispatches_through_single_workflow_router(tmp_path: Path, monkeypatch) -> None:
    from weiss_rl.workflows import runner

    args = SimpleNamespace(command="train-b1", repo_root=Path("ignored"))
    routed: dict[str, object] = {}

    def dispatch_workflow_command(*, args: object, repo_root: Path, python_exe: str) -> bool:
        routed["args"] = args
        routed["repo_root"] = repo_root
        routed["python_exe"] = python_exe
        return True

    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "_repo_root", lambda args_repo_root: tmp_path / "repo")
    monkeypatch.setattr(runner.sys, "executable", "python.exe")
    monkeypatch.setattr(runner, "dispatch_workflow_command", dispatch_workflow_command)

    runner.main()

    assert routed == {
        "args": args,
        "repo_root": tmp_path / "repo",
        "python_exe": "python.exe",
    }


def test_evaluation_parser_preserves_smoke_and_final_eval_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    smoke_args = build_parser().parse_args(
        [
            "smoke-eval",
            "--dry-run",
            "--run-dir",
            "runs/main_smoke",
            "--b1-run",
            "runs/b1",
        ]
    )
    final_args = build_parser().parse_args(
        [
            "eval-final",
            "--run-dir",
            "runs/main",
            "--b1-baseline-run-dir",
            "runs/b1_final",
        ]
    )

    assert smoke_args.command == "smoke-eval"
    assert smoke_args.dry_run is True
    assert smoke_args.run_dir == Path("runs/main_smoke")
    assert smoke_args.b1_baseline_run_dir == Path("runs/b1")
    assert final_args.command == "eval-final"
    assert final_args.run_dir == Path("runs/main")
    assert final_args.b1_baseline_run_dir == Path("runs/b1_final")


def test_evaluation_parser_preserves_smoke_eval_optional_b1_default() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(["smoke-eval", "--run-dir", "runs/main_smoke"])

    assert args.command == "smoke-eval"
    assert args.run_dir == Path("runs/main_smoke")
    assert args.b1_baseline_run_dir is None


def test_evaluation_parser_preserves_figures_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    default_args = build_parser().parse_args(["figures", "--run-dir", "runs/main"])
    override_args = build_parser().parse_args(
        [
            "figures",
            "--run-dir",
            "runs/main",
            "--fig-id",
            "winrate",
            "--format",
            "pdf",
            "--format",
            "png",
        ]
    )

    assert default_args.command == "figures"
    assert default_args.run_dir == Path("runs/main")
    assert default_args.fig_id == ""
    assert default_args.formats is None
    assert override_args.fig_id == "winrate"
    assert override_args.formats == ["pdf", "png"]


def test_evaluation_parser_preserves_b2_audit_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "b2-audit",
            "--run-dir",
            "runs/main",
            "--episodes-jsonl",
            "runs/main/eval/final_eval/episodes.jsonl",
            "--policy-id",
            "main_league_selected",
            "--output-run-dir",
            "runs/main/eval/b2_disagreement",
            "--snapshot-registry-json",
            "runs/main/training/snapshots/registry.json",
            "--summary-json",
            "runs/main/eval/b2_disagreement/summary.json",
            "--top-k",
            "37",
            "--top-actions",
            "8",
            "--allow-policy-id-mismatch",
            "--accept-snapshot-config-hash",
            "a" * 64,
            "--accept-snapshot-config-hash",
            "b" * 64,
        ]
    )

    assert args.command == "b2-audit"
    assert args.run_dir == Path("runs/main")
    assert args.episodes_jsonl == Path("runs/main/eval/final_eval/episodes.jsonl")
    assert args.policy_id == "main_league_selected"
    assert args.output_run_dir == Path("runs/main/eval/b2_disagreement")
    assert args.snapshot_registry_json == Path("runs/main/training/snapshots/registry.json")
    assert args.summary_json == Path("runs/main/eval/b2_disagreement/summary.json")
    assert args.top_k == 37
    assert args.top_actions == 8
    assert args.allow_policy_id_mismatch is True
    assert args.accept_snapshot_config_hash == ["a" * 64, "b" * 64]


def test_evaluation_parser_preserves_b2_audit_defaults() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "b2-audit",
            "--run-dir",
            "runs/main",
            "--episodes-jsonl",
            "runs/main/eval/final_eval/episodes.jsonl",
            "--policy-id",
            "main_league_selected",
        ]
    )

    assert args.output_run_dir is None
    assert args.snapshot_registry_json is None
    assert args.summary_json is None
    assert args.top_k == 25
    assert args.top_actions == 5
    assert args.allow_policy_id_mismatch is False
    assert args.accept_snapshot_config_hash == []


def test_training_parser_preserves_b1_and_guided_seed_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    b1_args = build_parser().parse_args(["train-b1", "--run-label", "b1_run"])
    guided_args = build_parser().parse_args(
        [
            "train-b1-guided-seed",
            "--dry-run",
            "--run-label",
            "guided_seed",
            "--profile",
            "gpu-probe",
        ]
    )

    assert b1_args.command == "train-b1"
    assert b1_args.run_label == "b1_run"
    assert b1_args.profile == "smoke"
    assert guided_args.command == "train-b1-guided-seed"
    assert guided_args.dry_run is True
    assert guided_args.run_label == "guided_seed"
    assert guided_args.profile == "gpu-probe"


def test_training_parser_preserves_main_training_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "train-main",
            "--run-label",
            "main_run",
            "--b1-baseline-run-dir",
            "runs/b1",
            "--seed-snapshot-run-dir",
            "runs/seed",
            "--init-policy-id",
            "selected_candidate",
            "--profile",
            "league-probe",
        ]
    )

    assert args.command == "train-main"
    assert args.run_label == "main_run"
    assert args.b1_baseline_run_dir == Path("runs/b1")
    assert args.seed_snapshot_run_dir == Path("runs/seed")
    assert args.init_policy_id == "selected_candidate"
    assert args.profile == "league-probe"


def test_training_parser_preserves_main_training_aliases_and_defaults() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(["train-main", "--run-label", "main_run", "--b1-run", "runs/b1"])

    assert args.b1_baseline_run_dir == Path("runs/b1")
    assert args.seed_snapshot_run_dir is None
    assert args.init_policy_id == "auto"
    assert args.profile == "smoke"


def test_training_parser_preserves_guided_bootstrap_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "train-main-guided-bootstrap",
            "--run-label",
            "guided_main",
            "--init-from-checkpoint",
            "runs/seed/training/checkpoints/checkpoint_25.pt",
            "--init-from-run-dir",
            "runs/init_source",
            "--init-policy-id",
            "guided_selected",
            "--seed-run",
            "runs/seed",
            "--b1-run",
            "runs/b1",
            "--vtrace-clamp",
            "--seed-champions",
            "--selected-seed-champion",
            "--profile",
            "gpu-probe",
        ]
    )

    assert args.command == "train-main-guided-bootstrap"
    assert args.run_label == "guided_main"
    assert args.init_from_checkpoint == Path("runs/seed/training/checkpoints/checkpoint_25.pt")
    assert args.init_from_run_dir == Path("runs/init_source")
    assert args.init_policy_id == "guided_selected"
    assert args.seed_snapshot_run_dir == Path("runs/seed")
    assert args.b1_baseline_run_dir == Path("runs/b1")
    assert args.vtrace_clamp is True
    assert args.seed_champions is True
    assert args.selected_seed_champion is True
    assert args.profile == "gpu-probe"


def test_training_parser_preserves_guided_bootstrap_defaults() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "train-main-guided-bootstrap",
            "--run-label",
            "guided_main",
            "--seed-snapshot-run-dir",
            "runs/seed",
        ]
    )

    assert args.init_from_checkpoint is None
    assert args.init_from_run_dir is None
    assert args.init_policy_id == ""
    assert args.seed_snapshot_run_dir == Path("runs/seed")
    assert args.b1_baseline_run_dir is None
    assert args.vtrace_clamp is False
    assert args.seed_champions is False
    assert args.selected_seed_champion is False
    assert args.profile == "smoke"


def test_controller_parser_preserves_guard_run_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "guard-run",
            "--dry-run",
            "--run-dir",
            "runs/main_probe",
            "--required-anchor",
            "B2 HeuristicPublic",
            "--required-anchor",
            "B3 HeuristicPublicAggro",
            "--min-latest-anchor-score",
            "0.51",
            "--max-latest-drop",
            "0.07",
            "--require-promotion-pass-after-attempts",
            "4",
            "--max-consecutive-promotion-failures",
            "5",
            "--max-vtrace-rho-p99",
            "33.5",
        ]
    )

    assert args.command == "guard-run"
    assert args.dry_run is True
    assert args.run_dir == Path("runs/main_probe")
    assert args.required_anchor == ["B2 HeuristicPublic", "B3 HeuristicPublicAggro"]
    assert args.min_latest_anchor_score == 0.51
    assert args.max_latest_drop == 0.07
    assert args.require_promotion_pass_after_attempts == 4
    assert args.max_consecutive_promotion_failures == 5
    assert args.max_vtrace_rho_p99 == 33.5


def test_controller_parser_preserves_guided_bootstrap_loop_defaults_and_overrides() -> None:
    from weiss_rl.workflows.parsers import build_parser
    from weiss_rl.workflows.training_commands import MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG

    default_args = build_parser().parse_args(["guided-bootstrap-loop", "--initial-run-dir", "runs/floor_selected"])

    assert default_args.command == "guided-bootstrap-loop"
    assert default_args.initial_run_dir == Path("runs/floor_selected")
    assert default_args.initial_policy_id == "guided_bootstrap_floor_selected"
    assert default_args.seed_run_dir is None
    assert default_args.run_prefix == "b1_guided_floor_segmented"
    assert default_args.stack_config == MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG
    assert default_args.alias_policy_id == "guided_bootstrap_floor_segmented_selected"
    assert default_args.segments == 4
    assert default_args.segment_updates == 25
    assert default_args.confirm_paired_seeds == 64
    assert default_args.stop_on_latest_falloff is False

    override_args = build_parser().parse_args(
        [
            "guided-bootstrap-loop",
            "--initial-run-dir",
            "runs/floor_selected",
            "--initial-policy-id",
            "selected_policy",
            "--seed-run-dir",
            "runs/seed",
            "--run-prefix",
            "floor_loop",
            "--stack-config",
            "configs/custom.yaml",
            "--alias-policy-id",
            "selected_alias",
            "--segments",
            "2",
            "--segment-updates",
            "13",
            "--confirm-paired-seeds",
            "21",
            "--stop-on-latest-falloff",
        ]
    )

    assert override_args.initial_policy_id == "selected_policy"
    assert override_args.seed_run_dir == Path("runs/seed")
    assert override_args.run_prefix == "floor_loop"
    assert override_args.stack_config == Path("configs/custom.yaml")
    assert override_args.alias_policy_id == "selected_alias"
    assert override_args.segments == 2
    assert override_args.segment_updates == 13
    assert override_args.confirm_paired_seeds == 21
    assert override_args.stop_on_latest_falloff is True


def test_controller_parser_preserves_guarded_league_arguments() -> None:
    from weiss_rl.workflows.parsers import build_parser

    args = build_parser().parse_args(
        [
            "guarded-league-bootstrap",
            "--init-from-checkpoint",
            "runs/seed/training/checkpoints/checkpoint_25.pt",
            "--seed-snapshot-run-dir",
            "runs/seed",
            "--run-prefix",
            "guarded_selected",
            "--stack-config",
            "configs/custom_main.yaml",
            "--segments",
            "3",
            "--segment-updates",
            "9",
            "--first-init-schedule-offset-updates",
            "2",
            "--confirm-paired-seeds",
            "65",
            "--publish-min-confirm-paired-seeds",
            "257",
            "--confirm-recent-candidate-count",
            "2",
            "--reference-summary-json",
            "runs/reference/summary.json",
            "--multiobjective-reference-summary-json",
            "runs/ref_a/summary.json",
            "--multiobjective-reference-summary-json",
            "runs/ref_b/summary.json",
            "--multiobjective-fixed-opponent",
            "B2 HeuristicPublic",
            "--learned-guard-opponent",
            "main_league_selected",
            "--min-learned-guard-mean",
            "0.61",
            "--min-learned-guard-reference-delta",
            "0.12",
            "--reference-label",
            "published",
            "--min-required-anchor-score",
            "0.56",
            "--max-reference-drop",
            "0.03",
            "--selected-alias-policy-id",
            "main_selected",
        ]
    )

    assert args.command == "guarded-league-bootstrap"
    assert args.init_from_checkpoint == Path("runs/seed/training/checkpoints/checkpoint_25.pt")
    assert args.seed_snapshot_run_dir == Path("runs/seed")
    assert args.run_prefix == "guarded_selected"
    assert args.stack_config == Path("configs/custom_main.yaml")
    assert args.segments == 3
    assert args.segment_updates == 9
    assert args.first_init_schedule_offset_updates == 2
    assert args.confirm_paired_seeds == 65
    assert args.publish_min_confirm_paired_seeds == 257
    assert args.confirm_recent_candidate_count == 2
    assert args.reference_summary_json == Path("runs/reference/summary.json")
    assert args.multiobjective_reference_summary_json == [
        Path("runs/ref_a/summary.json"),
        Path("runs/ref_b/summary.json"),
    ]
    assert args.multiobjective_fixed_opponent == ["B2 HeuristicPublic"]
    assert args.learned_guard_opponent == ["main_league_selected"]
    assert args.min_learned_guard_mean == 0.61
    assert args.min_learned_guard_reference_delta == 0.12
    assert args.reference_label == "published"
    assert args.min_required_anchor_score == 0.56
    assert args.max_reference_drop == 0.03
    assert args.selected_alias_policy_id == "main_selected"


def test_controller_workflow_plan_builder_preserves_default_guard_run_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.controller_plan import (
        ControllerWorkflowPlan,
        build_controller_workflow_plan,
        build_controller_workflow_plan_for_request,
        controller_workflow_request,
    )

    run_dir = tmp_path / "runs" / "main_probe"
    args = SimpleNamespace(
        command="guard-run",
        run_dir=run_dir,
        required_anchor=None,
        min_latest_anchor_score=0.45,
        max_latest_drop=0.05,
        require_promotion_pass_after_attempts=3,
        max_consecutive_promotion_failures=3,
        max_vtrace_rho_p99=25.0,
    )

    request = controller_workflow_request(args=args, repo_root=tmp_path, python_exe="python.exe")
    plan = build_controller_workflow_plan_for_request(request)
    legacy_plan = build_controller_workflow_plan(args=args, python_exe="python.exe")

    assert request.command == "guard-run"
    assert request.dry_run is False
    assert plan == legacy_plan
    assert isinstance(plan, ControllerWorkflowPlan)
    assert plan.plan_name == "main_probe_guard-run"
    assert plan.payload == {"workflow": "guard-run"}
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.diagnostics.learning_progress"]
    assert "--league-guard" in plan.command
    assert plan.command.count("--guard-required-anchor") == 3
    assert "B2 HeuristicPublic" in plan.command
    assert "B3 HeuristicPublicAggro" in plan.command
    assert "B4 HeuristicPublicControl" in plan.command
    assert "--guard-max-vtrace-rho-p99" in plan.command
    assert "25.0" in plan.command


def test_controller_workflow_plan_builder_preserves_guided_loop_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.controller_plan import build_controller_workflow_plan

    args = SimpleNamespace(
        command="guided-bootstrap-loop",
        initial_run_dir=tmp_path / "runs" / "guided_floor",
        initial_policy_id="guided_bootstrap_floor_selected",
        seed_run_dir=tmp_path / "runs" / "seed",
        run_prefix="floor_loop",
        stack_config=Path("configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml"),
        alias_policy_id="guided_bootstrap_floor_segmented_selected",
        segments=2,
        segment_updates=25,
        confirm_paired_seeds=64,
        stop_on_latest_falloff=True,
    )

    plan = build_controller_workflow_plan(args=args, python_exe="python.exe")

    assert plan is not None
    assert plan.plan_name == "floor_loop_guided-bootstrap-loop"
    assert plan.payload == {
        "workflow": "guided-bootstrap-loop",
        "initial_policy_id": "guided_bootstrap_floor_selected",
        "segments": 2,
        "segment_updates": 25,
        "confirm_paired_seeds": 64,
    }
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.experiments.segmented_b1_guided_bootstrap_entrypoint"]
    assert "--seed-run-dir" in plan.command
    assert (tmp_path / "runs" / "seed").as_posix() in plan.command
    assert "--stop-on-latest-falloff" in plan.command


def test_controller_workflow_plan_builder_rejects_invalid_guarded_league_counts(tmp_path: Path) -> None:
    from weiss_rl.workflows.controller_plan import build_controller_workflow_plan

    args = SimpleNamespace(
        command="guarded-league-bootstrap",
        init_from_checkpoint=tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt",
        seed_snapshot_run_dir=tmp_path / "runs" / "seed",
        run_prefix="guarded_selected",
        stack_config=Path("configs/thesis/main_league_guided_bootstrap_selected.yaml"),
        segments=2,
        segment_updates=10,
        first_init_schedule_offset_updates=-1,
        confirm_paired_seeds=64,
        publish_min_confirm_paired_seeds=256,
        confirm_recent_candidate_count=1,
        reference_summary_json=None,
        multiobjective_reference_summary_json=[],
        multiobjective_fixed_opponent=[],
        learned_guard_opponent=[],
        min_learned_guard_mean=0.5,
        min_learned_guard_reference_delta=0.0,
        reference_label="reference",
        min_required_anchor_score=0.5,
        max_reference_drop=0.04,
        selected_alias_policy_id="main_league_selected",
    )

    try:
        build_controller_workflow_plan(args=args, python_exe="python.exe")
    except SystemExit as exc:
        assert str(exc) == "--first-init-schedule-offset-updates must be >= 0"
    else:
        raise AssertionError("expected invalid guarded-league args to raise SystemExit")


def test_training_workflow_plan_builder_preserves_b1_command_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.training_plan import (
        TrainingWorkflowPlan,
        build_training_workflow_plan,
        build_training_workflow_plan_for_request,
        training_workflow_request,
    )

    args = SimpleNamespace(
        command="train-b1",
        run_label="b1_plan",
        profile="gpu-probe",
    )

    request = training_workflow_request(args=args, repo_root=tmp_path, python_exe="python.exe")
    plan = build_training_workflow_plan_for_request(request)
    legacy_plan = build_training_workflow_plan(args=args, repo_root=tmp_path, python_exe="python.exe")

    assert isinstance(plan, TrainingWorkflowPlan)
    assert request.command == "train-b1"
    assert request.dry_run is False
    assert legacy_plan == plan
    assert plan.plan_name == "b1_plan"
    assert plan.payload == {"workflow": "train-b1", "profile": "gpu-probe"}
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert "configs/thesis/b1_noleague.yaml" in plan.command
    assert "--device" in plan.command
    assert "cuda" in plan.command
    assert "training.profile_timers=true" in plan.command


def test_training_workflow_plan_builder_resolves_guided_bootstrap_registry_checkpoint(tmp_path: Path) -> None:
    from weiss_rl.workflows.training_plan import build_training_workflow_plan

    repo_root = tmp_path / "repo"
    source_run = repo_root / "runs" / "guided_source"
    checkpoint_path = source_run / "training" / "checkpoints" / "checkpoint_90.pt"
    registry_path = source_run / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "snapshots": [
                    {
                        "policy_id": "guided_bootstrap_selected",
                        "update": 90,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        command="train-main-guided-bootstrap",
        run_label="guided_plan",
        profile="smoke",
        init_from_checkpoint=None,
        init_from_run_dir=source_run,
        init_policy_id="guided_bootstrap_selected",
        seed_snapshot_run_dir=source_run,
        b1_baseline_run_dir=None,
        vtrace_clamp=False,
        seed_champions=False,
        selected_seed_champion=True,
    )

    plan = build_training_workflow_plan(args=args, repo_root=repo_root, python_exe="python.exe")

    assert plan is not None
    assert plan.plan_name == "guided_plan"
    assert plan.payload["workflow"] == "train-main-guided-bootstrap"
    assert plan.payload["selected_seed_champion"] is True
    assert plan.payload["init_policy_id"] == "guided_bootstrap_selected"
    assert (
        "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml" in plan.command
    )
    assert checkpoint_path.as_posix() in plan.command


def test_workflow_runner_module_keeps_public_cli_dry_run_surface(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.workflows.runner",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "runner_b1_smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "runner_b1_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert "configs/thesis/b1_noleague.yaml" in payload["command"]


def test_package_cli_train_b1_dry_run_uses_thesis_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert "configs/thesis/b1_noleague.yaml" in payload["command"]
    assert "--runtime-mode" in payload["command"]
    assert "train_async_fast" in payload["command"]
    assert "--profile" in payload["command"]
    assert "fast" in payload["command"]


def test_package_cli_train_b1_gpu_probe_uses_cuda_probe_shape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_gpu_probe",
            "--profile",
            "gpu-probe",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_gpu_probe.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert payload["profile"] == "gpu-probe"
    assert "--device" in payload["command"]
    assert "cuda" in payload["command"]
    assert "--num-envs" in payload["command"]
    assert "32" in payload["command"]
    assert "--unroll-length" in payload["command"]
    assert "16" in payload["command"]
    assert "training.profile_timers=true" in payload["command"]


def test_package_cli_train_b1_league_probe_uses_early_guard_shape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_league_probe",
            "--profile",
            "league-probe",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_league_probe.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert payload["profile"] == "league-probe"
    assert "--device" in payload["command"]
    assert "cuda" in payload["command"]
    assert "--num-envs" in payload["command"]
    assert "288" in payload["command"]
    assert "--unroll-length" in payload["command"]
    assert "64" in payload["command"]
    assert "--max-updates" in payload["command"]
    assert "50" in payload["command"]
    assert "--checkpoint-interval-updates" in payload["command"]
    assert "5" in payload["command"]
    assert "system.collection_backend=process" in payload["command"]
    assert "training.profile_timers=true" in payload["command"]


def test_package_cli_train_b1_guided_seed_uses_guided_seed_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1-guided-seed",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_guided_seed_smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "b1_guided_seed_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-b1-guided-seed"
    assert "configs/thesis/b1_guided_seed.yaml" in payload["command"]


def test_package_cli_train_main_requires_b1_and_uses_main_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    checkpoint_path = _write_cli_b1_source_run(b1_run)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_smoke",
            "--b1-run",
            str(b1_run),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "main_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-main"
    assert payload["init_policy_id"] == "selected_candidate"
    assert (
        "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
        in payload["command"]
    )
    assert b1_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_accepts_guided_seed_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    seed_run = repo_root / "runs" / "b1_guided_seed_smoke"
    checkpoint_path = _write_cli_b1_source_run(b1_run)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_seeded_smoke",
            "--b1-run",
            str(b1_run),
            "--seed-run",
            str(seed_run),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_seeded_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main"
    assert "--b1-baseline-run-dir" in payload["command"]
    assert b1_run.as_posix() in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_uses_seed_and_warmstart_without_strict_b1(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_smoke",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert "configs/thesis/main_league_guided_bootstrap.yaml" in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert init_checkpoint.as_posix() in payload["command"]
    assert "--b1-baseline-run-dir" not in payload["command"]


def test_package_cli_train_main_guided_bootstrap_accepts_optional_strict_b1_anchor(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    b1_run = repo_root / "runs" / "b1_noleague_candidate"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_with_b1",
            "--seed-run",
            str(seed_run),
            "--b1-run",
            str(b1_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_with_b1.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert "--b1-baseline-run-dir" in payload["command"]
    assert b1_run.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_vtrace_uses_clamped_stack(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_vtrace",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--vtrace-clamp",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_vtrace.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["vtrace_clamp"] is True
    assert "configs/thesis/main_league_guided_bootstrap_vtrace.yaml" in payload["command"]


def test_package_cli_train_main_guided_bootstrap_seed_champions_uses_seedchampion_stack(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_seedchampion",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--seed-champions",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_seedchampion.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["seed_champions"] is True
    assert payload["vtrace_clamp"] is False
    assert "configs/thesis/main_league_guided_bootstrap_seedchampion.yaml" in payload["command"]


def test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_run = repo_root / "runs" / "guided_source"
    checkpoint_path = source_run / "training" / "checkpoints" / "checkpoint_90.pt"
    registry_path = source_run / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": "guided_bootstrap_selected",
                        "update": 90,
                        "weights_sha256": "a" * 64,
                        "path": "training/snapshots/guided_bootstrap_selected/weights.pt",
                    }
                ],
                "champion_snapshots": [],
                "pinned_snapshots": ["guided_bootstrap_selected"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_selected",
            "--seed-run",
            str(source_run),
            "--init-from-run-dir",
            str(source_run),
            "--init-policy-id",
            "guided_bootstrap_selected",
            "--selected-seed-champion",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "main_guided_selected.json").read_text())
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["selected_seed_champion"] is True
    assert payload["init_policy_id"] == "guided_bootstrap_selected"
    assert (
        "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
        in payload["command"]
    )
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_rejects_ambiguous_init_sources(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "ambiguous",
            "--seed-run",
            str(repo_root / "runs" / "seed"),
            "--init-from-checkpoint",
            str(repo_root / "runs" / "source" / "training" / "checkpoints" / "checkpoint_90.pt"),
            "--init-from-run-dir",
            str(repo_root / "runs" / "source"),
            "--init-policy-id",
            "guided_bootstrap_selected",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--init-from-checkpoint cannot be combined" in result.stderr


def test_evaluation_workflow_plan_builder_preserves_smoke_eval_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.evaluation_plan import (
        EvaluationWorkflowPlan,
        build_evaluation_workflow_plan,
        build_evaluation_workflow_plan_for_request,
        evaluation_workflow_request,
    )

    run_dir = tmp_path / "runs" / "main_smoke"
    b1_run = tmp_path / "runs" / "b1_anchor"
    args = SimpleNamespace(
        command="smoke-eval",
        run_dir=run_dir,
        b1_baseline_run_dir=b1_run,
    )

    request = evaluation_workflow_request(args=args, repo_root=tmp_path, python_exe="python.exe")
    plan = build_evaluation_workflow_plan_for_request(request)
    legacy_plan = build_evaluation_workflow_plan(args=args, python_exe="python.exe")

    assert request.command == "smoke-eval"
    assert request.dry_run is False
    assert plan == legacy_plan
    assert isinstance(plan, EvaluationWorkflowPlan)
    assert plan.plan_name == "main_smoke_smoke-eval"
    assert plan.payload == {"workflow": "smoke-eval"}
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.workflows.eval_entrypoint"]
    assert "configs/thesis/final_eval.yaml" in plan.command
    assert "--b1-baseline-run-dir" in plan.command
    assert b1_run.as_posix() in plan.command
    assert plan.command.count("--policy-id") == 5
    assert "--skip-readiness" in plan.command


def test_evaluation_workflow_plan_builder_preserves_figures_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.evaluation_plan import build_evaluation_workflow_plan

    run_dir = tmp_path / "runs" / "main_smoke"
    args = SimpleNamespace(
        command="figures",
        run_dir=run_dir,
        fig_id="seat_bias",
        formats=["png", "pdf"],
    )

    plan = build_evaluation_workflow_plan(args=args, python_exe="python.exe")

    assert plan is not None
    assert plan.plan_name == "main_smoke_figures"
    assert plan.payload == {"workflow": "figures"}
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.workflows.figures_entrypoint"]
    assert "--fig-id" in plan.command
    assert "seat_bias" in plan.command
    assert plan.command.count("--format") == 2
    assert "png" in plan.command
    assert "pdf" in plan.command


def test_evaluation_workflow_plan_builder_preserves_b2_audit_shape(tmp_path: Path) -> None:
    from weiss_rl.workflows.evaluation_plan import build_evaluation_workflow_plan

    run_dir = tmp_path / "runs" / "main_smoke"
    episodes_jsonl = run_dir / "eval" / "final_eval" / "episodes.jsonl"
    snapshot_registry_json = run_dir / "training" / "snapshots" / "registry.json"
    summary_json = run_dir / "eval" / "b2_disagreement" / "summary.json"
    args = SimpleNamespace(
        command="b2-audit",
        run_dir=run_dir,
        episodes_jsonl=episodes_jsonl,
        policy_id="policy_000001",
        output_run_dir=None,
        snapshot_registry_json=snapshot_registry_json,
        summary_json=summary_json,
        top_k=11,
        top_actions=3,
        allow_policy_id_mismatch=True,
        accept_snapshot_config_hash=["abc123"],
    )

    plan = build_evaluation_workflow_plan(args=args, python_exe="python.exe")

    assert plan is not None
    assert plan.plan_name == "main_smoke_b2-audit"
    assert plan.payload == {"workflow": "b2-audit"}
    assert plan.command[:3] == ["python.exe", "-m", "weiss_rl.diagnostics.b2_disagreement_audit"]
    assert "--output-run-dir" in plan.command
    assert (run_dir / "eval" / "b2_disagreement").as_posix() in plan.command
    assert "--snapshot-registry-json" in plan.command
    assert snapshot_registry_json.as_posix() in plan.command
    assert "--summary-json" in plan.command
    assert summary_json.as_posix() in plan.command
    assert "--allow-policy-id-mismatch" in plan.command
    assert "--accept-snapshot-config-hash" in plan.command
    assert "abc123" in plan.command


def test_package_cli_smoke_eval_uses_tiny_eval_budget(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "smoke-eval",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(repo_root / "runs" / "main_smoke"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_smoke_smoke-eval.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "smoke-eval"
    assert "configs/thesis/final_eval.yaml" in payload["command"]
    assert payload["command"].count("--policy-id") == 5
    assert "B4 HeuristicPublicControl" in payload["command"]
    assert "--paired-seed-limit" in payload["command"]
    assert "--skip-readiness" in payload["command"]


def test_package_cli_figures_wraps_package_figure_entrypoint(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_smoke"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "figures",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--fig-id",
            "seat_bias",
            "--format",
            "png",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_smoke_figures.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "figures"
    assert "-m" in payload["command"]
    assert "weiss_rl.workflows.figures_entrypoint" in payload["command"]
    assert "--run-dir" in payload["command"]
    assert run_dir.as_posix() in payload["command"]
    assert "--fig-id" in payload["command"]
    assert "seat_bias" in payload["command"]
    assert "--format" in payload["command"]
    assert "png" in payload["command"]


def test_workflow_figures_command_builder_preserves_optional_arguments(tmp_path: Path) -> None:
    from weiss_rl.workflows.evaluation_commands import _figures_command

    run_dir = tmp_path / "runs" / "main_smoke"
    command = _figures_command(
        python_exe="python.exe",
        run_dir=run_dir,
        fig_id="seat_bias",
        formats=("png", "pdf"),
    )

    assert command[:3] == ["python.exe", "-m", "weiss_rl.workflows.figures_entrypoint"]
    assert "--run-dir" in command
    assert run_dir.as_posix() in command
    assert "--fig-id" in command
    assert "seat_bias" in command
    assert command.count("--format") == 2
    assert "png" in command
    assert "pdf" in command


def test_package_cli_b2_audit_wraps_standard_disagreement_module(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_smoke"
    episodes_jsonl = run_dir / "eval" / "final_eval" / "episodes.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "b2-audit",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--episodes-jsonl",
            str(episodes_jsonl),
            "--policy-id",
            "policy_000001",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_smoke_b2-audit.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "b2-audit"
    assert "-m" in payload["command"]
    assert "weiss_rl.diagnostics.b2_disagreement_audit" in payload["command"]
    assert "configs/thesis/final_eval.yaml" in payload["command"]
    assert "--episodes-jsonl" in payload["command"]
    assert episodes_jsonl.as_posix() in payload["command"]
    assert "--policy-id" in payload["command"]
    assert "policy_000001" in payload["command"]
    assert (run_dir / "eval" / "b2_disagreement").as_posix() in payload["command"]


def test_package_cli_guard_run_wraps_learning_progress_league_guard(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_probe"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guard-run",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--min-latest-anchor-score",
            "0.5",
            "--max-vtrace-rho-p99",
            "25",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_probe_guard-run.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "guard-run"
    assert "-m" in payload["command"]
    assert "weiss_rl.diagnostics.learning_progress" in payload["command"]
    assert "--league-guard" in payload["command"]
    assert "--run-dir" in payload["command"]
    assert run_dir.as_posix() in payload["command"]
    assert "--guard-min-latest-anchor-score" in payload["command"]
    assert "0.5" in payload["command"]
    assert "--guard-max-vtrace-rho-p99" in payload["command"]
    assert "25.0" in payload["command"]
    assert payload["command"].count("--guard-required-anchor") == 3
    assert "B4 HeuristicPublicControl" in payload["command"]


def test_package_cli_guided_bootstrap_loop_wraps_segmented_controller(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    initial_run_dir = repo_root / "runs" / "guided_floor"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guided-bootstrap-loop",
            "--repo-root",
            str(repo_root),
            "--initial-run-dir",
            str(initial_run_dir),
            "--initial-policy-id",
            "guided_bootstrap_floor_selected",
            "--run-prefix",
            "floor_loop",
            "--segments",
            "2",
            "--segment-updates",
            "25",
            "--confirm-paired-seeds",
            "64",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "floor_loop_guided-bootstrap-loop.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "guided-bootstrap-loop"
    assert "-m" in payload["command"]
    assert "weiss_rl.experiments.segmented_b1_guided_bootstrap_entrypoint" in payload["command"]
    assert "--initial-run-dir" in payload["command"]
    assert initial_run_dir.as_posix() in payload["command"]
    assert "--stack-config" in payload["command"]
    assert "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml" in payload["command"]
    assert "--confirm-paired-seeds" in payload["command"]
    assert "64" in payload["command"]


def test_package_cli_guarded_league_bootstrap_wraps_controller(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "guided_selected"
    init_checkpoint = seed_run / "training" / "checkpoints" / "checkpoint_25.pt"
    reference_summary = seed_run / "eval" / "targeted_confirm256" / "targeted_confirm256_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guarded-league-bootstrap",
            "--repo-root",
            str(repo_root),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--seed-snapshot-run-dir",
            str(seed_run),
            "--run-prefix",
            "guarded_selected",
            "--segments",
            "2",
            "--segment-updates",
            "10",
            "--confirm-paired-seeds",
            "64",
            "--publish-min-confirm-paired-seeds",
            "256",
            "--confirm-recent-candidate-count",
            "3",
            "--first-init-schedule-offset-updates",
            "0",
            "--reference-summary-json",
            str(reference_summary),
            "--reference-label",
            "selected_confirm256",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "guarded_selected_guarded-league-bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["workflow"] == "guarded-league-bootstrap"
    assert "-m" in payload["command"]
    assert "weiss_rl.experiments.guarded_league_bootstrap_entrypoint" in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert init_checkpoint.as_posix() in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--reference-summary-json" in payload["command"]
    assert reference_summary.as_posix() in payload["command"]
    assert "--first-init-schedule-offset-updates" in payload["command"]
    assert "0" in payload["command"]
    assert "--publish-min-confirm-paired-seeds" in payload["command"]
    assert "--confirm-recent-candidate-count" in payload["command"]
    assert "3" in payload["command"]
    assert "--max-reference-drop" in payload["command"]
    assert "0.04" in payload["command"]
    assert "--selected-alias-policy-id" in payload["command"]
    assert "main_league_selected" in payload["command"]
    assert payload["publish_min_confirm_paired_seeds"] == 256
    assert payload["confirm_recent_candidate_count"] == 3
    assert payload["selected_alias_policy_id"] == "main_league_selected"


def test_package_cli_guard_run_failure_exits_without_traceback(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "main_bad"
    logs_dir = run_dir / "training" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "training_metrics.jsonl").write_text(
        json.dumps({"update_count": 20, "loss": 1.0, "vtrace_rho_p99": 31.0}) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "periodic_dev_eval_summaries.json").write_text(
        json.dumps(
            {
                "train_u20_p4": {
                    "update_count": 20,
                    "aggregate_score": 0.50,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.34,
                        "B3 HeuristicPublicAggro": 0.41,
                        "B4 HeuristicPublicControl": 0.47,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    for update in (10, 15, 20):
        gate_path = run_dir / "eval" / "promotion_gate" / f"update_{update}" / "promotion_gate.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(
            json.dumps(
                {
                    "focal_policy_id": f"policy_{update:06d}",
                    "decision": {"passed": False, "reasons": [{"code": "anchor_loss_guardrail_exceeded"}]},
                    "overall_posterior": {"mean": 0.5, "prob_gt_target": 0.1},
                }
            ),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guard-run",
            "--run-dir",
            str(run_dir),
            "--max-vtrace-rho-p99",
            "25",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "league guard failed" in result.stderr
    assert "Traceback" not in result.stderr
