from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn


def maybe_compile_learner_model(
    *,
    model: Any,
    training_config: Any,
    device: torch.device,
    compile_fn: Callable[..., nn.Module] | None = None,
    log_fn: Callable[[str], None] = print,
) -> nn.Module | None:
    if not bool(getattr(training_config, "compile_learner", False)):
        return None
    if device.type != "cuda":
        log_fn(
            "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
        )
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if callable(enable_trunk_compile):
            try:
                enable_trunk_compile(mode="reduce-overhead")
            except Exception as exc:
                log_fn(f"Learner compile note: structured trunk compile failed; skipping torch.compile ({exc!r}).")
                return None
            log_fn("Enabled torch.compile for the structured learner trunk (mode=reduce-overhead).")
            return model
        log_fn(
            "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; "
            "skipping torch.compile."
        )
        return None
    compile_model = torch.compile if compile_fn is None else compile_fn
    compiled = compile_model(model, mode="reduce-overhead")
    log_fn("Enabled torch.compile for the learner forward path (mode=reduce-overhead).")
    return cast(nn.Module, compiled)
