"""Learner construction/setup helpers used by the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn


def format_compile_learner_not_cuda_message() -> str:
    return (
        "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
    )


def format_compile_learner_trunk_failed_message(exception_repr: str) -> str:
    return f"Learner compile note: structured trunk compile failed; skipping torch.compile ({exception_repr})."


def format_compile_learner_trunk_enabled_message() -> str:
    return "Enabled torch.compile for the structured learner trunk (mode=reduce-overhead)."


def format_compile_learner_missing_trunk_hook_message() -> str:
    return "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; skipping torch.compile."


def format_compile_learner_forward_enabled_message() -> str:
    return "Enabled torch.compile for the learner forward path (mode=reduce-overhead)."


def format_trainable_main_residual_policy_enabled_message(
    *,
    checkpoint_path: Path,
    alpha: float,
    hidden_dim: int,
    residual_mode: str,
    initial_state_path_text: str,
) -> str:
    return (
        "Enabled trainable main residual policy: "
        f"base={checkpoint_path} alpha={float(alpha):g} "
        f"hidden_dim={int(hidden_dim)} "
        f"mode={residual_mode} "
        f"initial_state={initial_state_path_text or '<zero>'}"
    )


def maybe_compile_learner_model(
    *,
    model: nn.Module,
    training_config: Any,
    device: torch.device,
    compile_fn: Callable[..., nn.Module] | None = None,
    emit: Callable[[str], None] = print,
) -> nn.Module | None:
    """Optionally compile the learner model while preserving train.py's public notices."""

    if not bool(getattr(training_config, "compile_learner", False)):
        return None
    if device.type != "cuda":
        emit(format_compile_learner_not_cuda_message())
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if callable(enable_trunk_compile):
            try:
                enable_trunk_compile(mode="reduce-overhead")
            except Exception as exc:
                emit(format_compile_learner_trunk_failed_message(repr(exc)))
                return None
            emit(format_compile_learner_trunk_enabled_message())
            return model
        emit(format_compile_learner_missing_trunk_hook_message())
        return None
    compile_model = torch.compile if compile_fn is None else compile_fn
    compiled = compile_model(model, mode="reduce-overhead")
    emit(format_compile_learner_forward_enabled_message())
    return compiled
