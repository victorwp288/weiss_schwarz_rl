from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.model import PolicyValueModel
from weiss_rl.training.dev_eval import clone_cpu_eval_model


def _model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=8,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
    )


def test_clone_cpu_eval_model_copies_weights_guidance_and_eval_mode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del tmp_path
    stack = SimpleNamespace(config=SimpleNamespace(model=_model_config()))
    learner_model = PolicyValueModel(observation_dim=4, action_dim=3, config=stack.config.model)
    learner_model.train()
    guidance_calls: list[tuple[PolicyValueModel, dict[str, float]]] = []
    guidance_payload = {
        "public_heuristic_logit_bias_scale": 0.25,
        "public_heuristic_actor_logit_bias_scale": 0.75,
    }
    monkeypatch.setattr(
        "weiss_rl.training.dev_eval.model_clone.model_guidance_payload",
        lambda model: guidance_payload if model is learner_model else {},
    )
    monkeypatch.setattr(
        "weiss_rl.training.dev_eval.model_clone.restore_model_guidance_from_payload",
        lambda model, payload: guidance_calls.append((model, dict(payload))),
    )

    clone = clone_cpu_eval_model(
        learner_model=learner_model,
        observation_dim=4,
        action_dim=3,
        stack=stack,
    )

    assert clone.training is False
    assert guidance_calls == [(clone, guidance_payload)]
    for source, copied in zip(learner_model.parameters(), clone.parameters(), strict=True):
        assert copied.device.type == "cpu"
        assert torch.equal(copied, source.detach().cpu())
        assert copied.data_ptr() != source.detach().cpu().data_ptr()


def test_clone_cpu_eval_model_requires_model_config() -> None:
    learner_model = PolicyValueModel(observation_dim=4, action_dim=3, config=_model_config())
    stack = SimpleNamespace(config=SimpleNamespace(model=None))

    with pytest.raises(RuntimeError, match="missing the model config block"):
        clone_cpu_eval_model(
            learner_model=learner_model,
            observation_dim=4,
            action_dim=3,
            stack=stack,
        )
