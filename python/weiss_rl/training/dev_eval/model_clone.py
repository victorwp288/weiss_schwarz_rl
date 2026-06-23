"""Model cloning for deterministic CPU dev-eval."""

from __future__ import annotations

from typing import Any

import torch

from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.training.guidance import model_guidance_payload, restore_model_guidance_from_payload


def clone_cpu_eval_model(
    *,
    learner_model: PolicyValueModel,
    observation_dim: int,
    action_dim: int,
    stack: Any,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    """Clone the learner model into an eval-mode CPU model for deterministic eval."""

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    cpu_state_dict = {name: value.detach().cpu().clone() for name, value in learner_model.state_dict().items()}
    eval_model.load_state_dict(cpu_state_dict)
    restore_model_guidance_from_payload(eval_model, model_guidance_payload(learner_model))
    eval_model.eval()
    return eval_model


__all__ = [
    "clone_cpu_eval_model",
]
