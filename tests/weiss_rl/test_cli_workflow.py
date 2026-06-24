from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.workflows.command_surface import (
    PUBLIC_THESIS_COMMANDS,
    public_workflow_command,
    public_workflow_command_payload,
    public_workflow_commands_for_group,
)
from weiss_rl.workflows.evaluation_workflow.plan import EVALUATION_PLAN_BUILDERS, build_evaluation_workflow_plan
from weiss_rl.workflows.parsers import build_parser
from weiss_rl.workflows.training_workflow.plan import TRAINING_PLAN_BUILDERS, build_training_workflow_plan
from weiss_rl.workflows.workflow_route_explanation import (
    public_workflow_lifecycle_payload,
    public_workflow_route_rows,
    render_public_workflow_route_summary,
)


def _subcommands() -> set[str]:
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if getattr(action, "dest", None) == "command")
    return set(subparsers_action.choices)


def test_parser_exposes_only_lean_thesis_workflow_commands() -> None:
    assert _subcommands() == set(PUBLIC_THESIS_COMMANDS)


def test_public_workflow_command_registry_names_command_groups() -> None:
    assert tuple(command.name for command in public_workflow_commands_for_group("training")) == (
        "train-b1",
        "train-main",
    )
    assert tuple(command.name for command in public_workflow_commands_for_group("evaluation")) == (
        "smoke-eval",
        "eval-final",
        "figures",
        "b2-audit",
    )
    train_main = public_workflow_command("train-main")
    assert train_main.group == "training"
    assert train_main.evidence_role == "produces candidate main checkpoints for final evaluation"
    assert public_workflow_command_payload(train_main)["next_step"] == (
        "run eval-final on the retained main run with the selected B1 anchor"
    )


def test_public_workflow_command_help_includes_inputs_and_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train-main", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "Train the main league policy from an explicit retained B1 anchor." in help_text
    assert "Evidence role:" in help_text
    assert "candidate main checkpoints" in help_text
    assert "Inputs:" in help_text
    assert "selected B1 run" in help_text
    assert "Outputs:" in help_text
    assert "main policy snapshots" in help_text
    assert "Next step:" in help_text
    assert "run eval-final" in help_text


def test_public_workflow_route_explanation_matches_plan_builders() -> None:
    rows = public_workflow_route_rows()
    lifecycle = public_workflow_lifecycle_payload()

    assert tuple(row.command for row in rows) == tuple(PUBLIC_THESIS_COMMANDS)
    assert [step["step_id"] for step in lifecycle] == [
        "register_command",
        "parse_arguments",
        "build_plan",
        "dispatch",
        "retain_outputs",
    ]
    for row in rows:
        if row.group == "training":
            assert row.command in TRAINING_PLAN_BUILDERS
            assert row.plan_builder == TRAINING_PLAN_BUILDERS[row.command].__name__
            assert row.dispatch_target == "dispatch_training_request"
        elif row.group == "evaluation":
            assert row.command in EVALUATION_PLAN_BUILDERS
            assert row.plan_builder == EVALUATION_PLAN_BUILDERS[row.command].__name__
            assert row.dispatch_target == "dispatch_evaluation_request"
        else:
            raise AssertionError(f"unexpected command group: {row.group}")

    summary = render_public_workflow_route_summary()
    assert "train-main | training | produces candidate main checkpoints for final evaluation" in summary
    assert "dispatch_training_request" in summary
    assert "eval-final | evaluation | produces thesis-grade policy-panel evidence" in summary
    assert "dispatch_evaluation_request" in summary
    assert "selected B1 run" in summary
    assert "readiness artifacts" in summary
    assert "check paper-readiness and then export figures" in summary


def test_train_main_workflow_uses_report_retained_main_config(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-main",
            "--run-label",
            "main_smoke",
            "--b1-run",
            str(tmp_path / "b1"),
            "--profile",
            "smoke",
            "--dry-run",
        ]
    )

    b1_registry = tmp_path / "b1" / "training" / "snapshots"
    b1_registry.mkdir(parents=True)
    checkpoint_dir = tmp_path / "b1" / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_1.pt").write_bytes(b"weights")
    (b1_registry / "registry.json").write_text(
        """
{
  "snapshots": [
    {
      "policy_id": "selected_candidate",
      "path": "training/snapshots/selected_candidate/weights.pt",
      "update": 1
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    plan = build_training_workflow_plan(args=args, repo_root=tmp_path, python_exe="python.exe")

    assert plan is not None
    assert plan.payload["workflow"] == "train-main"
    assert plan.payload["evidence_role"] == "produces candidate main checkpoints for final evaluation"
    assert plan.payload["next_step"] == "run eval-final on the retained main run with the selected B1 anchor"
    assert (
        plan.payload["profile_purpose"]
        == "fast wiring check for commands, configs, simulator import, and artifact paths"
    )
    assert plan.payload["profile_evidence_level"] == "plumbing_only"
    assert plan.payload["num_envs"] == 2
    assert plan.payload["max_updates"] == 1
    assert plan.payload["checkpoint_interval_updates"] == 1
    assert [stage["name"] for stage in plan.payload["workflow_stages"]] == [
        "select_profile",
        "load_stack_config",
        "resolve_seed_policy",
        "run_training_entrypoint",
        "retain_evidence",
    ]
    assert plan.payload["evidence_targets"] == [
        "manifest.json",
        "run_summary.json",
        "determinism_report.json",
        "training/logs/training_metrics.jsonl",
        "training/checkpoints/checkpoint_tracker.json",
        "training/snapshots/registry.json",
        "training/logs/periodic_dev_eval_summaries.json",
        "eval/promotion_gate/",
    ]
    assert "configs/thesis/main_league.yaml" in plan.command
    assert "main_league_guided_bootstrap" not in plan.command


def test_train_b1_workflow_payload_lists_retained_evidence_targets() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-b1",
            "--run-label",
            "b1_smoke",
            "--profile",
            "smoke",
            "--dry-run",
        ]
    )

    plan = build_training_workflow_plan(args=args, repo_root=Path(), python_exe="python.exe")

    assert plan is not None
    assert plan.payload["workflow"] == "train-b1"
    assert plan.payload["profile_evidence_level"] == "plumbing_only"
    assert plan.payload["workflow_stages"][-1]["evidence"] == [
        "manifest",
        "metrics log",
        "checkpoint tracker",
        "snapshot registry",
        "periodic dev eval",
    ]
    assert plan.payload["evidence_targets"] == [
        "manifest.json",
        "run_summary.json",
        "determinism_report.json",
        "training/logs/training_metrics.jsonl",
        "training/checkpoints/checkpoint_tracker.json",
        "training/snapshots/registry.json",
    ]


def test_smoke_eval_workflow_payload_explains_public_eval_command(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "smoke-eval",
            "--run-dir",
            str(tmp_path / "main"),
            "--b1-run",
            str(tmp_path / "b1"),
            "--dry-run",
        ]
    )

    plan = build_evaluation_workflow_plan(args=args, python_exe="python.exe")

    assert plan is not None
    assert plan.payload["workflow"] == "smoke-eval"
    assert plan.payload["workflow_purpose"] == "Run the small fixed-panel evaluation used to check plumbing."
    assert plan.payload["evidence_role"] == "checks evaluation wiring only; it is not model-quality evidence"
    assert plan.payload["next_step"] == "use eval-final for retained thesis evidence"
    assert plan.payload["inputs"] == ("main run directory", "optional B1 run")
    assert plan.payload["outputs"] == ("smoke evaluation summary", "small episode record")
    assert plan.payload["run_dir"] == (tmp_path / "main").as_posix()
    assert plan.payload["b1_baseline_run_dir"] == (tmp_path / "b1").as_posix()
    assert plan.payload["smoke"] is True
    assert [stage["name"] for stage in plan.payload["workflow_stages"]] == [
        "resolve_command",
        "resolve_policy_sources",
        "run_evaluation_entrypoint",
        "write_evidence",
    ]
    assert plan.payload["evidence_targets"] == [
        "eval/final_eval/summary.json",
        "eval/final_eval/episodes.jsonl",
    ]


def test_eval_final_workflow_payload_lists_retained_evidence_targets(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-final",
            "--run-dir",
            str(tmp_path / "main"),
            "--b1-run",
            str(tmp_path / "b1"),
            "--dry-run",
        ]
    )

    plan = build_evaluation_workflow_plan(args=args, python_exe="python.exe")

    assert plan.payload["workflow"] == "eval-final"
    assert plan.payload["workflow_stages"][1]["evidence"] == [
        "snapshot registry",
        "selected policy ids",
        "B1 baseline run",
    ]
    assert plan.payload["evidence_targets"] == [
        "eval/final_eval/summary.json",
        "eval/final_eval/matchups.csv",
        "eval/final_eval/matrices/mean.csv",
        "paper_readiness_summary.json",
    ]


def test_removed_workflow_commands_are_rejected_by_parser() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train-main-guided-bootstrap", "--run-label", "old"])

    with pytest.raises(SystemExit):
        parser.parse_args(["guard-run", "--run-dir", "runs/old"])
