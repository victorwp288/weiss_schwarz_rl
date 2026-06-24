"""Workflow-stage names shared by evaluation dry-run plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationWorkflowStage:
    name: str
    purpose: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "evidence": list(self.evidence),
        }


EVALUATION_WORKFLOW_STAGES: tuple[EvaluationWorkflowStage, ...] = (
    EvaluationWorkflowStage(
        name="resolve_command",
        purpose="Choose smoke eval, final eval, figures, or B2 audit behavior from the public command.",
        evidence=("workflow", "run_dir", "command arguments"),
    ),
    EvaluationWorkflowStage(
        name="resolve_policy_sources",
        purpose="Resolve the main run, optional B1 run, and policy ids needed by the evaluation command.",
        evidence=("snapshot registry", "selected policy ids", "B1 baseline run"),
    ),
    EvaluationWorkflowStage(
        name="run_evaluation_entrypoint",
        purpose="Dispatch the concrete evaluation, figure, or diagnostic entrypoint.",
        evidence=("command", "seed contract", "policy panel"),
    ),
    EvaluationWorkflowStage(
        name="write_evidence",
        purpose="Write the retained summaries, matrices, figures, or diagnostic files expected by the workflow.",
        evidence=("summary files", "episode records", "matrices", "readiness or diagnostic outputs"),
    ),
)


def evaluation_workflow_stage_payload() -> list[dict[str, object]]:
    return [stage.as_payload() for stage in EVALUATION_WORKFLOW_STAGES]


__all__ = [
    "EVALUATION_WORKFLOW_STAGES",
    "EvaluationWorkflowStage",
    "evaluation_workflow_stage_payload",
]
