"""Help text rendering for public workflow commands."""

from __future__ import annotations

from weiss_rl.workflows.command_surface import PublicWorkflowCommand


def public_workflow_command_epilog(command: PublicWorkflowCommand) -> str:
    inputs = "\n".join(f"  - {item}" for item in command.inputs)
    outputs = "\n".join(f"  - {item}" for item in command.outputs)
    return (
        f"Evidence role:\n  {command.evidence_role}\n\n"
        f"Inputs:\n{inputs}\n\n"
        f"Outputs:\n{outputs}\n\n"
        f"Next step:\n  {command.next_step}"
    )


__all__ = ["public_workflow_command_epilog"]
