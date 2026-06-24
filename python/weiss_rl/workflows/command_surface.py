"""Named public thesis workflow commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WorkflowCommandGroup = Literal["training", "evaluation"]


@dataclass(frozen=True, slots=True)
class PublicWorkflowCommand:
    name: str
    group: WorkflowCommandGroup
    help: str
    description: str
    evidence_role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    next_step: str


TRAIN_B1_COMMAND = PublicWorkflowCommand(
    name="train-b1",
    group="training",
    help="Train the B1 NoLeague baseline",
    description="Train the no-league B1 baseline from a named thesis profile.",
    evidence_role="produces the retained baseline anchor used by main training and final evaluation",
    inputs=("run label", "training profile"),
    outputs=("run directory under runs/", "training metrics", "snapshot registry"),
    next_step="inspect the run, then pass the retained B1 run to train-main",
)
TRAIN_MAIN_COMMAND = PublicWorkflowCommand(
    name="train-main",
    group="training",
    help="Train the main league thesis model",
    description="Train the main league policy from an explicit retained B1 anchor.",
    evidence_role="produces candidate main checkpoints for final evaluation",
    inputs=("run label", "selected B1 run", "training profile"),
    outputs=("run directory under runs/", "training metrics", "main policy snapshots"),
    next_step="run eval-final on the retained main run with the selected B1 anchor",
)
SMOKE_EVAL_COMMAND = PublicWorkflowCommand(
    name="smoke-eval",
    group="evaluation",
    help="Run a tiny deterministic eval on a run directory",
    description="Run the small fixed-panel evaluation used to check plumbing.",
    evidence_role="checks evaluation wiring only; it is not model-quality evidence",
    inputs=("main run directory", "optional B1 run"),
    outputs=("smoke evaluation summary", "small episode record"),
    next_step="use eval-final for retained thesis evidence",
)
EVAL_FINAL_COMMAND = PublicWorkflowCommand(
    name="eval-final",
    group="evaluation",
    help="Run the thesis-grade final evaluation",
    description="Run the retained final-evaluation panel for a thesis run.",
    evidence_role="produces thesis-grade policy-panel evidence and readiness artifacts",
    inputs=("main run directory", "selected B1 run"),
    outputs=("final evaluation summary", "paired-seed matchup records", "readiness artifacts"),
    next_step="check paper-readiness and then export figures",
)
FIGURES_COMMAND = PublicWorkflowCommand(
    name="figures",
    group="evaluation",
    help="Export paper figures and tables for a run",
    description="Export retained paper figures and tables from an evaluated run.",
    evidence_role="renders retained evaluation artifacts into paper-facing outputs",
    inputs=("run directory", "optional figure id", "optional output formats"),
    outputs=("paper figure files", "paper table files"),
    next_step="compare rendered outputs against the retained evaluation summary",
)
B2_AUDIT_COMMAND = PublicWorkflowCommand(
    name="b2-audit",
    group="evaluation",
    help="Run the standard learner-vs-B2 disagreement audit",
    description="Compare learner choices with the B2 public heuristic on retained episodes.",
    evidence_role="diagnoses where the learned policy disagrees with the B2 public heuristic",
    inputs=("run directory", "episodes JSONL", "policy id"),
    outputs=("B2 disagreement summary", "top disagreement examples"),
    next_step="use the disagreement summary to inspect replay states and policy alignment",
)

PUBLIC_WORKFLOW_COMMANDS = (
    TRAIN_B1_COMMAND,
    TRAIN_MAIN_COMMAND,
    SMOKE_EVAL_COMMAND,
    EVAL_FINAL_COMMAND,
    FIGURES_COMMAND,
    B2_AUDIT_COMMAND,
)
PUBLIC_WORKFLOW_COMMAND_BY_NAME = {command.name: command for command in PUBLIC_WORKFLOW_COMMANDS}
PUBLIC_THESIS_COMMANDS = tuple(command.name for command in PUBLIC_WORKFLOW_COMMANDS)


def public_workflow_command(name: str) -> PublicWorkflowCommand | None:
    return PUBLIC_WORKFLOW_COMMAND_BY_NAME.get(str(name))


def public_workflow_commands_for_group(group: WorkflowCommandGroup | str) -> tuple[PublicWorkflowCommand, ...]:
    resolved_group = str(group)
    return tuple(command for command in PUBLIC_WORKFLOW_COMMANDS if command.group == resolved_group)


def public_workflow_command_payload(command: PublicWorkflowCommand) -> dict[str, object]:
    return {
        "workflow": command.name,
        "workflow_purpose": command.description,
        "evidence_role": command.evidence_role,
        "inputs": command.inputs,
        "outputs": command.outputs,
        "next_step": command.next_step,
    }


__all__ = [
    "B2_AUDIT_COMMAND",
    "EVAL_FINAL_COMMAND",
    "FIGURES_COMMAND",
    "PUBLIC_WORKFLOW_COMMAND_BY_NAME",
    "PUBLIC_THESIS_COMMANDS",
    "PUBLIC_WORKFLOW_COMMANDS",
    "PublicWorkflowCommand",
    "SMOKE_EVAL_COMMAND",
    "TRAIN_B1_COMMAND",
    "TRAIN_MAIN_COMMAND",
    "WorkflowCommandGroup",
    "public_workflow_command_payload",
    "public_workflow_command",
    "public_workflow_commands_for_group",
]
