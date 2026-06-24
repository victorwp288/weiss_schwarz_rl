"""Named outputs shared by the structured trunk and policy head."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredTrunkOutputField:
    name: str
    shape_role: str
    consumer: str

    def as_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "shape_role": self.shape_role,
            "consumer": self.consumer,
        }


STRUCTURED_TRUNK_OUTPUT_CONTRACT: tuple[StructuredTrunkOutputField, ...] = (
    StructuredTrunkOutputField(
        name="recurrent_output",
        shape_role="row-wise latent state for the acting seat",
        consumer="value head, packed/factorized policy helpers, diagnostics",
    ),
    StructuredTrunkOutputField(
        name="state_repr",
        shape_role="policy-head state representation built from recurrent output and observation context",
        consumer="structured legal-action scoring",
    ),
    StructuredTrunkOutputField(
        name="observation_context",
        shape_role="typed observation slices and public board summaries",
        consumer="candidate features and public-heuristic bias",
    ),
    StructuredTrunkOutputField(
        name="value",
        shape_role="scalar value estimate aligned to the input row or sequence step",
        consumer="learner bootstrap, evaluation summaries",
    ),
    StructuredTrunkOutputField(
        name="next_seat_hidden",
        shape_role="seat-aware recurrent state after the acting-seat update",
        consumer="runtime rollout state, sequence continuation",
    ),
)


def structured_trunk_output_contract_payload() -> list[dict[str, str]]:
    return [field.as_payload() for field in STRUCTURED_TRUNK_OUTPUT_CONTRACT]


__all__ = [
    "STRUCTURED_TRUNK_OUTPUT_CONTRACT",
    "StructuredTrunkOutputField",
    "structured_trunk_output_contract_payload",
]
