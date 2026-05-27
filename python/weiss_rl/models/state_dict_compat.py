"""Compatibility helpers for optional model extensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

OPPONENT_CONTEXT_TRAINABLE_STATE_KEYS = frozenset(
    {
        "opponent_context_action_bias_adapter",
        "opponent_context_candidate_residual_context",
        "opponent_context_candidate_residual_candidate.weight",
        "opponent_context_candidate_residual_meta.weight",
        "opponent_context_candidate_residual_out.weight",
        "opponent_context_candidate_residual_state.weight",
        "opponent_context_hidden_adapter",
        "opponent_context_recurrent_adapter",
    }
)


def compatible_missing_model_state_keys(expected_state_dict: Mapping[str, Any]) -> set[str]:
    """Return optional keys old checkpoints may omit for the active model."""

    return {
        key
        for key in OPPONENT_CONTEXT_TRAINABLE_STATE_KEYS
        if key in expected_state_dict and isinstance(expected_state_dict[key], torch.Tensor)
    }


def state_dict_key_mismatch_for_context_compat(
    *,
    source_state_dict: Mapping[str, Any],
    expected_state_dict: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    source_keys = set(source_state_dict)
    expected_keys = set(expected_state_dict)
    missing = expected_keys - source_keys
    extra = source_keys - expected_keys
    allowed_missing = compatible_missing_model_state_keys(expected_state_dict)
    return sorted(missing - allowed_missing), sorted(extra), sorted(missing & allowed_missing)


def load_model_state_dict_with_context_compat(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    context: str,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load a model state dict, allowing only the zero-init context adapter to be absent."""

    if not hasattr(model, "state_dict"):
        return model.load_state_dict(state_dict)

    expected_state_dict = model.state_dict()
    disallowed_missing, extra, allowed_missing = state_dict_key_mismatch_for_context_compat(
        source_state_dict=state_dict,
        expected_state_dict=expected_state_dict,
    )
    if extra or disallowed_missing:
        return model.load_state_dict(state_dict)
    if not allowed_missing:
        return model.load_state_dict(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = sorted(str(key) for key in result.unexpected_keys)
    missing = sorted(str(key) for key in result.missing_keys)
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(
            f"{context} model state_dict compatibility load failed: missing_keys={missing} unexpected_keys={unexpected}"
        )
    return result


__all__ = [
    "OPPONENT_CONTEXT_TRAINABLE_STATE_KEYS",
    "compatible_missing_model_state_keys",
    "load_model_state_dict_with_context_compat",
    "state_dict_key_mismatch_for_context_compat",
]
