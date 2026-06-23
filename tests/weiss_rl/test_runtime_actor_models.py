from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from weiss_rl.runtime.components.policy_inference.actor_models import (
    actor_inference_model,
    maybe_compile_runtime_actor_model,
)


class _CompileStructuredActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.compile_calls = 0
        self.compile_mode: str | None = None

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> _CompileStructuredActorModel:
        self.compile_calls += 1
        self.compile_mode = mode
        return self


def test_actor_model_compile_helper_enables_structured_actor_trunk_compile() -> None:
    model = _CompileStructuredActorModel()

    compiled = maybe_compile_runtime_actor_model(cast(Any, model), enabled=True)

    assert compiled is model
    assert model.compile_calls == 1
    assert model.compile_mode == "reduce-overhead"


def test_actor_inference_model_prefers_compiled_model() -> None:
    model = object()
    compiled_model = object()
    actor = SimpleNamespace(model=model, compiled_model=compiled_model)

    assert actor_inference_model(cast(Any, actor)) is compiled_model

    actor.compiled_model = None
    assert actor_inference_model(cast(Any, actor)) is model
