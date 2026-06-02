from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch

from weiss_rl.runtime import _actor_inference_model, _maybe_compile_runtime_actor_model
from weiss_rl.runtime.components.actor_models import actor_inference_model, maybe_compile_runtime_actor_model


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


def test_actor_model_compile_helper_matches_runtime_wrapper() -> None:
    direct_model = _CompileStructuredActorModel()
    wrapper_model = _CompileStructuredActorModel()

    direct_compiled = maybe_compile_runtime_actor_model(cast(Any, direct_model), enabled=True)
    wrapper_compiled = _maybe_compile_runtime_actor_model(cast(Any, wrapper_model), enabled=True)

    assert _maybe_compile_runtime_actor_model is not maybe_compile_runtime_actor_model
    assert direct_compiled is direct_model
    assert wrapper_compiled is wrapper_model
    assert direct_model.compile_mode == "reduce-overhead"
    assert wrapper_model.compile_mode == "reduce-overhead"


def test_actor_inference_model_prefers_compiled_model_and_preserves_wrapper() -> None:
    model = object()
    compiled_model = object()
    actor = SimpleNamespace(model=model, compiled_model=compiled_model)

    assert actor_inference_model(cast(Any, actor)) is compiled_model
    assert _actor_inference_model(cast(Any, actor)) is compiled_model
    assert _actor_inference_model is not actor_inference_model

    actor.compiled_model = None
    assert actor_inference_model(cast(Any, actor)) is model
    assert _actor_inference_model(cast(Any, actor)) is model
