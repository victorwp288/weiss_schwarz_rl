"""Factory for dense and structured policy/value models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.config.models import ModelConfig
from weiss_rl.models.backbone.base import GLOBAL_ACTION_SPACE_SIZE, STRUCTURED_V2_ENCODER_KIND
from weiss_rl.models.policy.policy_value_model import PolicyValueModel
from weiss_rl.models.policy.structured_policy_value_model import StructuredLegalPolicyValueModel


@dataclass(frozen=True, slots=True)
class PolicyValueFactoryRoute:
    route_id: str
    condition: str
    model_class: str
    required_inputs: tuple[str, ...]
    purpose: str

    def as_payload(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "condition": self.condition,
            "model_class": self.model_class,
            "required_inputs": list(self.required_inputs),
            "purpose": self.purpose,
        }


POLICY_VALUE_FACTORY_ROUTES: tuple[PolicyValueFactoryRoute, ...] = (
    PolicyValueFactoryRoute(
        route_id="structured_v2",
        condition="config.encoder_kind == 'structured_v2'",
        model_class="StructuredLegalPolicyValueModel",
        required_inputs=("observation_dim", "config", "action_dim", "observation_spec", "spec_bundle"),
        purpose="Build the thesis model with typed observation context and simulator legal-action scoring.",
    ),
    PolicyValueFactoryRoute(
        route_id="dense_fallback",
        condition="all other encoder kinds",
        model_class="PolicyValueModel",
        required_inputs=("observation_dim", "config", "action_dim"),
        purpose="Build the dense compatibility model with a flat policy head over the global action catalog.",
    ),
)


def policy_value_factory_route_payload() -> list[dict[str, object]]:
    return [route.as_payload() for route in POLICY_VALUE_FACTORY_ROUTES]


def build_policy_value_model(
    *,
    observation_dim: int,
    config: ModelConfig,
    action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
    dropout_p: float | None = None,
    observation_spec: Mapping[str, Any] | None = None,
    spec_bundle: Mapping[str, Any] | None = None,
    card_table: Mapping[str, Any] | None = None,
) -> PolicyValueModel:
    """Build the dense or structured policy/value model from the model config."""

    encoder_kind = str(config.encoder_kind).strip().lower()
    if encoder_kind == STRUCTURED_V2_ENCODER_KIND:
        return StructuredLegalPolicyValueModel(
            observation_dim=observation_dim,
            config=config,
            action_dim=action_dim,
            dropout_p=dropout_p,
            observation_spec=observation_spec,
            spec_bundle=spec_bundle,
            card_table=card_table,
        )
    return PolicyValueModel(
        observation_dim=observation_dim,
        config=config,
        action_dim=action_dim,
        dropout_p=dropout_p,
        observation_spec=observation_spec,
    )


__all__ = [
    "POLICY_VALUE_FACTORY_ROUTES",
    "PolicyValueFactoryRoute",
    "build_policy_value_model",
    "policy_value_factory_route_payload",
]
