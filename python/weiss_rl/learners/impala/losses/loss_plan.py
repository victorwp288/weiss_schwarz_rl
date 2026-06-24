"""Named IMPALA loss components and the evidence each one uses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImpalaLossComponentPlan:
    name: str
    purpose: str
    owner_modules: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "owner_modules": list(self.owner_modules),
            "evidence": list(self.evidence),
        }


IMPALA_LOSS_COMPONENT_PLAN: tuple[ImpalaLossComponentPlan, ...] = (
    ImpalaLossComponentPlan(
        name="vtrace_targets",
        purpose="Resolve off-policy returns and policy-gradient advantages from behavior/current log-probs.",
        owner_modules=("weiss_rl.learners.impala.losses.loss_vtrace_stage", "weiss_rl.learners.vtrace_torch"),
        evidence=("behavior log-prob", "current action log-prob", "rho/c clip settings", "rewards", "discounts"),
    ),
    ImpalaLossComponentPlan(
        name="policy_gradient",
        purpose="Increase probability of sampled actions with positive V-trace advantage.",
        owner_modules=("weiss_rl.learners.impala.losses.objective_loss",),
        evidence=("action log-prob", "V-trace advantages", "loss mask"),
    ),
    ImpalaLossComponentPlan(
        name="value_regression",
        purpose="Fit the value head to V-trace targets on configured value-train rows.",
        owner_modules=("weiss_rl.learners.impala.losses.objective_loss", "weiss_rl.learners.impala.losses.loss_objective_stage"),
        evidence=("values", "V-trace targets", "value train mask", "value loss coefficient"),
    ),
    ImpalaLossComponentPlan(
        name="entropy_bonus",
        purpose="Keep exploration pressure at the configured candidate or family entropy scope.",
        owner_modules=("weiss_rl.learners.impala.losses.objective_loss", "weiss_rl.learners.packed_action_logp"),
        evidence=("entropy tensor", "entropy coefficient", "entropy scope"),
    ),
    ImpalaLossComponentPlan(
        name="trajectory_retention",
        purpose="Optionally imitate retained trajectory actions without adding rows to policy-gradient loss.",
        owner_modules=("weiss_rl.learners.trajectory_retention", "weiss_rl.learners.impala.losses.objective_loss"),
        evidence=("trajectory retention mask", "retention action log-prob", "retention coefficient"),
    ),
    ImpalaLossComponentPlan(
        name="policy_anchor",
        purpose="Optionally regularize the learner against an anchor policy distribution.",
        owner_modules=("weiss_rl.learners.impala.losses.loss_policy_anchor_stage", "weiss_rl.learners.policy_anchor"),
        evidence=("anchor policy", "anchor logits/log-probs", "anchor coefficients"),
    ),
    ImpalaLossComponentPlan(
        name="teacher_auxiliary",
        purpose="Optionally train structured action-family, argument, margin, and public-heuristic targets.",
        owner_modules=("weiss_rl.learners.impala.losses.loss_teacher_stage", "weiss_rl.learners.structured_teacher"),
        evidence=("teacher labels", "packed legal view", "factorized outputs", "teacher coefficients"),
    ),
    ImpalaLossComponentPlan(
        name="structured_metrics",
        purpose="Emit structured policy diagnostics without changing the objective.",
        owner_modules=("weiss_rl.learners.impala.support.metrics_assembly", "weiss_rl.learners.impala.support.structured_summary"),
        evidence=("structured metrics mode", "action catalog", "packed legal view", "factorized result"),
    ),
)


def impala_loss_component_plan_payload() -> list[dict[str, object]]:
    return [component.as_payload() for component in IMPALA_LOSS_COMPONENT_PLAN]


__all__ = [
    "IMPALA_LOSS_COMPONENT_PLAN",
    "ImpalaLossComponentPlan",
    "impala_loss_component_plan_payload",
]
