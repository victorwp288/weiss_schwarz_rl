"""Read-only map from public workflow commands to their plan builders."""

from __future__ import annotations

from dataclasses import dataclass

from weiss_rl.workflows.command_surface import PUBLIC_WORKFLOW_COMMANDS
from weiss_rl.workflows.evaluation_workflow.plan import EVALUATION_PLAN_BUILDERS
from weiss_rl.workflows.training_workflow.plan import TRAINING_PLAN_BUILDERS


@dataclass(frozen=True, slots=True)
class WorkflowRouteRow:
    command: str
    group: str
    evidence_role: str
    dispatch_target: str
    plan_builder: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    next_step: str


@dataclass(frozen=True, slots=True)
class PublicWorkflowLifecycleStep:
    step_id: str
    purpose: str
    owner_modules: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "purpose": self.purpose,
            "owner_modules": list(self.owner_modules),
        }


PUBLIC_WORKFLOW_LIFECYCLE: tuple[PublicWorkflowLifecycleStep, ...] = (
    PublicWorkflowLifecycleStep(
        step_id="register_command",
        purpose="Declare the public thesis command, evidence role, inputs, outputs, and next step.",
        owner_modules=("weiss_rl.workflows.command_surface",),
    ),
    PublicWorkflowLifecycleStep(
        step_id="parse_arguments",
        purpose="Expose only the lean thesis command set through the CLI parser.",
        owner_modules=(
            "weiss_rl.workflows.parsers",
            "weiss_rl.workflows.training_workflow.parser",
            "weiss_rl.workflows.evaluation_workflow.parser",
        ),
    ),
    PublicWorkflowLifecycleStep(
        step_id="build_plan",
        purpose="Resolve command arguments into a dry-run payload and concrete command invocation.",
        owner_modules=("weiss_rl.workflows.training_workflow.plan", "weiss_rl.workflows.evaluation_workflow.plan"),
    ),
    PublicWorkflowLifecycleStep(
        step_id="dispatch",
        purpose="Route the request to the training or evaluation dispatcher.",
        owner_modules=("weiss_rl.workflows.workflow_dispatch",),
    ),
    PublicWorkflowLifecycleStep(
        step_id="retain_outputs",
        purpose="Write the run, evaluation, figure, or diagnostic artifacts named by the command payload.",
        owner_modules=(
            "weiss_rl.workflows.training_workflow.dispatch",
            "weiss_rl.workflows.evaluation_workflow.dispatch",
        ),
    ),
)


def public_workflow_lifecycle_payload() -> list[dict[str, object]]:
    return [step.as_payload() for step in PUBLIC_WORKFLOW_LIFECYCLE]


def public_workflow_route_rows() -> tuple[WorkflowRouteRow, ...]:
    rows: list[WorkflowRouteRow] = []
    for command in PUBLIC_WORKFLOW_COMMANDS:
        if command.group == "training":
            builder = TRAINING_PLAN_BUILDERS[command.name]
            dispatch_target = "dispatch_training_request"
        elif command.group == "evaluation":
            builder = EVALUATION_PLAN_BUILDERS[command.name]
            dispatch_target = "dispatch_evaluation_request"
        else:
            raise AssertionError(f"Unhandled workflow command group: {command.group}")
        rows.append(
            WorkflowRouteRow(
                command=command.name,
                group=command.group,
                evidence_role=command.evidence_role,
                dispatch_target=dispatch_target,
                plan_builder=builder.__name__,
                inputs=command.inputs,
                outputs=command.outputs,
                next_step=command.next_step,
            )
        )
    return tuple(rows)


def render_public_workflow_route_summary() -> str:
    lines = ["Command | Group | Evidence role | Dispatch | Plan builder | Inputs | Outputs | Next step"]
    lines.append("--- | --- | --- | --- | --- | --- | --- | ---")
    for row in public_workflow_route_rows():
        lines.append(
            " | ".join(
                (
                    row.command,
                    row.group,
                    row.evidence_role,
                    row.dispatch_target,
                    row.plan_builder,
                    ", ".join(row.inputs),
                    ", ".join(row.outputs),
                    row.next_step,
                )
            )
        )
    return "\n".join(lines)


__all__ = [
    "WorkflowRouteRow",
    "PUBLIC_WORKFLOW_LIFECYCLE",
    "PublicWorkflowLifecycleStep",
    "public_workflow_lifecycle_payload",
    "public_workflow_route_rows",
    "render_public_workflow_route_summary",
]
