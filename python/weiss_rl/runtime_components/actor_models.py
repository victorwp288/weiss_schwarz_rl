"""Actor model selection helpers for the training runtime."""

from __future__ import annotations

from typing import Any

import torch


def maybe_compile_runtime_actor_model(model: Any, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if not callable(enable_trunk_compile):
            return None
        try:
            enable_trunk_compile(mode="reduce-overhead")
        except Exception:
            return None
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return None


def actor_inference_model(actor: Any) -> Any:
    return actor.compiled_model if actor.compiled_model is not None else actor.model
