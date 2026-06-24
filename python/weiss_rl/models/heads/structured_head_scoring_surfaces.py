"""Named scoring surfaces exposed by the structured legal-action head."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredHeadScoringSurface:
    name: str
    purpose: str
    entrypoints: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "entrypoints": list(self.entrypoints),
            "evidence": list(self.evidence),
        }


STRUCTURED_HEAD_SCORING_SURFACES: tuple[StructuredHeadScoringSurface, ...] = (
    StructuredHeadScoringSurface(
        name="dense_legal_logits",
        purpose="Return full action-catalog logits while masking illegal actions.",
        entrypoints=("score_legal_actions", "forward"),
        evidence=("legal mask or packed ids", "negative fill value", "global action ids"),
    ),
    StructuredHeadScoringSurface(
        name="packed_candidate_logits",
        purpose="Score only simulator-provided legal candidates for runtime and learner efficiency.",
        entrypoints=("score_packed_candidates", "forward_packed_seat_aware", "sample_packed_seat_aware"),
        evidence=("packed ids", "packed offsets", "candidate metadata"),
    ),
    StructuredHeadScoringSurface(
        name="factorized_policy",
        purpose="Expose action-family and argument distributions for structured supervision and diagnostics.",
        entrypoints=("evaluate_factorized_packed", "evaluate_factorized_sequence_packed_seat_aware"),
        evidence=("family logits", "argument logits", "same-family reference actions"),
    ),
    StructuredHeadScoringSurface(
        name="public_heuristic_bias",
        purpose="Score transparent public-board preferences used as optional bias or teacher signal.",
        entrypoints=("score_packed_public_heuristic_candidates",),
        evidence=("public heuristic profile", "slot preferences", "candidate families"),
    ),
)


def structured_head_scoring_surface_payload() -> list[dict[str, object]]:
    return [surface.as_payload() for surface in STRUCTURED_HEAD_SCORING_SURFACES]


__all__ = [
    "STRUCTURED_HEAD_SCORING_SURFACES",
    "StructuredHeadScoringSurface",
    "structured_head_scoring_surface_payload",
]
