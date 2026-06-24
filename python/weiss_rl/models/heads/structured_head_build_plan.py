"""Build-order checklist for the structured legal-action head."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredHeadBuildStep:
    step_id: str
    title: str
    purpose: str


STRUCTURED_HEAD_BUILD_PLAN = (
    StructuredHeadBuildStep(
        step_id="validate_inputs",
        title="Validate dimensions and bias scales",
        purpose="Reject invalid model dimensions, chunk sizes, and public-heuristic bias scales before modules exist.",
    ),
    StructuredHeadBuildStep(
        step_id="resolve_blueprint",
        title="Resolve catalog blueprint",
        purpose="Build the catalog view, action component tables, feature dimensions, offsets, and factorized lookups.",
    ),
    StructuredHeadBuildStep(
        step_id="install_catalog_view",
        title="Install catalog view and offsets",
        purpose="Expose family IDs, attack type IDs, public-bias family IDs, and candidate feature offsets.",
    ),
    StructuredHeadBuildStep(
        step_id="install_representation_modules",
        title="Install representation modules",
        purpose="Create embeddings, projections, slot encoders, state projection, candidate projection, and joint scorer.",
    ),
    StructuredHeadBuildStep(
        step_id="install_action_tables",
        title="Install action tables",
        purpose="Register dense action-family and argument lookup buffers.",
    ),
    StructuredHeadBuildStep(
        step_id="install_factorized_modules",
        title="Install factorized modules",
        purpose="Create family and argument query heads plus factorized lookup buffers.",
    ),
    StructuredHeadBuildStep(
        step_id="install_public_heuristic_buffers",
        title="Install public-heuristic buffers",
        purpose="Register slot preferences used by optional public-heuristic logit biasing.",
    ),
    StructuredHeadBuildStep(
        step_id="set_runtime_chunking",
        title="Set runtime chunking",
        purpose="Record CPU/GPU candidate scoring chunk sizes and factorized row chunk sizes.",
    ),
)


__all__ = ["STRUCTURED_HEAD_BUILD_PLAN", "StructuredHeadBuildStep"]
