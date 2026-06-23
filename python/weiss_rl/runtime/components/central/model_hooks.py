"""Lazy model hook helpers for central actor-row forwarding."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

from weiss_rl.runtime.components.policy_inference.actor_models import (
    actor_inference_model as resolve_actor_inference_model,
)

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def actor_inference_model(actor: _ActorState) -> Any:
    return resolve_actor_inference_model(actor)


def model_accepts_legal_actions_kwarg(model: Any) -> bool:
    target = getattr(model, "_orig_mod", model)
    forward = getattr(target, "forward_seat_aware", None)
    if forward is None:
        return False
    try:
        parameters = signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if "legal_actions" in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


__all__ = ["actor_inference_model", "model_accepts_legal_actions_kwarg"]
