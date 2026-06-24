"""Named periodic dev-eval stages and their evidence outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalPlanStep:
    step_id: str
    purpose: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "purpose": self.purpose,
            "evidence": list(self.evidence),
        }


PERIODIC_DEV_EVAL_PLAN: tuple[PeriodicDevEvalPlanStep, ...] = (
    PeriodicDevEvalPlanStep(
        step_id="validate_contract",
        purpose="Validate that dev-eval settings, seed sources, and runtime assumptions are usable.",
        evidence=("evaluation contract", "seed file", "validated seed sources"),
    ),
    PeriodicDevEvalPlanStep(
        step_id="snapshot_eval_model",
        purpose="Clone the current learner into a CPU eval model tied to the current checkpoint.",
        evidence=("checkpoint path", "policy version", "observation/action dimensions"),
    ),
    PeriodicDevEvalPlanStep(
        step_id="resolve_anchor_panel",
        purpose="Build the fixed opponent panel for this update.",
        evidence=("opponent policy ids", "display names", "heuristic/model sources"),
    ),
    PeriodicDevEvalPlanStep(
        step_id="run_matchups",
        purpose="Run paired-seed games against every anchor and write per-matchup artifacts.",
        evidence=("paired seeds", "matchup summaries", "uncertainty statistics"),
    ),
    PeriodicDevEvalPlanStep(
        step_id="summarize_quality",
        purpose="Aggregate anchor scores into the update-level dev-eval summary.",
        evidence=("aggregate score", "anchor scores", "primary matchup summary"),
    ),
    PeriodicDevEvalPlanStep(
        step_id="attach_diagnostics",
        purpose="Attach stall and policy-alignment diagnostics when the matchup exposes them.",
        evidence=("stall monitor", "policy alignment diagnostics", "summary.json"),
    ),
)


def periodic_dev_eval_plan_payload() -> list[dict[str, object]]:
    return [step.as_payload() for step in PERIODIC_DEV_EVAL_PLAN]


__all__ = [
    "PERIODIC_DEV_EVAL_PLAN",
    "PeriodicDevEvalPlanStep",
    "periodic_dev_eval_plan_payload",
]
