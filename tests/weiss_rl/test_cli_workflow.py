from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.workflows.parsers import build_parser
from weiss_rl.workflows.training_workflow.plan import build_training_workflow_plan

PUBLIC_CLI_COMMANDS = {
    "train-b1",
    "train-main",
    "smoke-eval",
    "eval-final",
    "figures",
    "b2-audit",
}


def _subcommands() -> set[str]:
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if getattr(action, "dest", None) == "command")
    return set(subparsers_action.choices)


def test_parser_exposes_only_lean_thesis_workflow_commands() -> None:
    assert _subcommands() == PUBLIC_CLI_COMMANDS


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
    assert "configs/thesis/main_league.yaml" in plan.command
    assert "main_league_guided_bootstrap" not in plan.command


def test_removed_workflow_commands_are_rejected_by_parser() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train-main-guided-bootstrap", "--run-label", "old"])

    with pytest.raises(SystemExit):
        parser.parse_args(["guard-run", "--run-dir", "runs/old"])
