"""Reader-facing map of the policy/value model components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelArchitectureComponent:
    key: str
    role: str
    owner_modules: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "role": self.role,
            "owner_modules": list(self.owner_modules),
            "evidence": list(self.evidence),
        }


MODEL_ARCHITECTURE_COMPONENTS: tuple[ModelArchitectureComponent, ...] = (
    ModelArchitectureComponent(
        key="factory",
        role="Chooses dense fallback versus structured legal-action model construction.",
        owner_modules=("weiss_rl.models.policy.policy_value_factory",),
        evidence=("model config", "simulator spec bundle", "action catalog"),
    ),
    ModelArchitectureComponent(
        key="dense_trunk",
        role="Encodes observations, applies the recurrent or feedforward core, and predicts value.",
        owner_modules=(
            "weiss_rl.models.policy.policy_value_model",
            "weiss_rl.models.backbone.base",
            "weiss_rl.models.backbone.policy_value_recurrent",
            "weiss_rl.models.backbone.policy_value_trunk",
            "weiss_rl.models.backbone.trunk_contract",
        ),
        evidence=("observation_dim", "recurrent_core", "value_head"),
    ),
    ModelArchitectureComponent(
        key="opponent_context",
        role="Adds opponent-indexed hidden offsets, action bias, and candidate residuals.",
        owner_modules=("weiss_rl.models.policy.opponent_context_mixin",),
        evidence=("opponent policy ids", "opponent context indices", "bias/residual config"),
    ),
    ModelArchitectureComponent(
        key="structured_head",
        role="Scores legal simulator candidates through packed, factorized, and dense views.",
        owner_modules=(
            "weiss_rl.models.heads.structured_head",
            "weiss_rl.models.heads.structured_head_blueprint",
            "weiss_rl.models.heads.structured_head_build_plan",
            "weiss_rl.models.heads.structured_head_scoring_surfaces",
        ),
        evidence=("action catalog", "candidate features", "legal masks", "build plan", "scoring surfaces"),
    ),
    ModelArchitectureComponent(
        key="public_heuristic_bias",
        role="Optionally biases legal candidates with transparent public-board heuristics.",
        owner_modules=(
            "weiss_rl.models.public_heuristic.public_heuristic_scoring",
            "weiss_rl.models.public_heuristic.public_heuristic_attack_scoring",
            "weiss_rl.models.public_heuristic.public_heuristic_bias",
        ),
        evidence=("public heuristic profiles", "family ids", "attack/slot preferences"),
    ),
    ModelArchitectureComponent(
        key="diagnostics",
        role="Summarizes structured action scoring and teacher-supervision behavior.",
        owner_modules=(
            "weiss_rl.models.scoring.factorized_diagnostics",
            "weiss_rl.learners.structured_policy_metrics",
        ),
        evidence=("top action summaries", "same-family metrics", "teacher auxiliary metrics"),
    ),
)


def model_architecture_component_payload() -> list[dict[str, object]]:
    return [component.as_payload() for component in MODEL_ARCHITECTURE_COMPONENTS]


__all__ = [
    "MODEL_ARCHITECTURE_COMPONENTS",
    "ModelArchitectureComponent",
    "model_architecture_component_payload",
]
