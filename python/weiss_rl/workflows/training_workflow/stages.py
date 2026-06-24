"""Workflow-stage names shared by training dry-run plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingWorkflowStage:
    name: str
    purpose: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "evidence": list(self.evidence),
        }


TRAINING_WORKFLOW_STAGES: tuple[TrainingWorkflowStage, ...] = (
    TrainingWorkflowStage(
        name="select_profile",
        purpose="Resolve the profile into environment count, unroll length, device, and update budget.",
        evidence=("profile", "num_envs", "unroll_length", "max_updates", "device"),
    ),
    TrainingWorkflowStage(
        name="load_stack_config",
        purpose="Choose the B1 no-league stack or the main league stack.",
        evidence=("stack config path", "runtime mode", "simulator profile"),
    ),
    TrainingWorkflowStage(
        name="resolve_seed_policy",
        purpose="For main training, resolve the retained B1 checkpoint used to initialize the run.",
        evidence=("b1 run", "init policy id", "init checkpoint path"),
    ),
    TrainingWorkflowStage(
        name="run_training_entrypoint",
        purpose="Dispatch the train entrypoint with explicit profile-derived command arguments.",
        evidence=("run label", "command", "config overrides"),
    ),
    TrainingWorkflowStage(
        name="retain_evidence",
        purpose="Keep the artifacts needed to inspect learning quality and run final evaluation.",
        evidence=("manifest", "metrics log", "checkpoint tracker", "snapshot registry", "periodic dev eval"),
    ),
)


def training_workflow_stage_payload() -> list[dict[str, object]]:
    return [stage.as_payload() for stage in TRAINING_WORKFLOW_STAGES]


__all__ = [
    "TRAINING_WORKFLOW_STAGES",
    "TrainingWorkflowStage",
    "training_workflow_stage_payload",
]
