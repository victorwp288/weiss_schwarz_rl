"""Terminal step decoding helpers for evaluation results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

MISSING = object()


def winner_seat_from_terminal_step(
    step: object,
    *,
    env_index: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    acting_seat: int | None,
) -> int | None:
    if not terminated or truncated:
        return None

    explicit_winner_seat = optional_terminal_winner_seat(step, env_index=env_index)
    if explicit_winner_seat is not MISSING:
        return cast(int | None, explicit_winner_seat)
    if reward == 0.0:
        return None

    perspective_seat = reward_perspective_seat(step, env_index=env_index, acting_seat=acting_seat)
    return perspective_seat if reward > 0.0 else 1 - perspective_seat


def optional_terminal_winner_seat(step: object, *, env_index: int) -> object:
    for name in ("winner_seat", "winner"):
        if not hasattr(step, name):
            continue
        value = step_value_for_env(step, name=name, env_index=env_index)
        if value is None:
            return None

        seat = int(value)
        if seat == -1:
            return None
        return require_seat(seat, name=name)
    return MISSING


def require_seat(value: int, *, name: str) -> int:
    seat = int(value)
    if seat not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {seat}")
    return seat


def step_value_for_env(step: object, *, name: str, env_index: int) -> Any:
    values = np.asarray(getattr(step, name), dtype=object)
    if values.ndim == 0:
        value = values.item()
    else:
        value = values[env_index]
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def step_scalar(
    step: object,
    names: Sequence[str],
    *,
    env_index: int,
    cast_fn: Any,
) -> Any:
    for name in names:
        if hasattr(step, name):
            return cast_fn(step_value_for_env(step, name=name, env_index=env_index))
    joined_names = ", ".join(names)
    raise AttributeError(f"step is missing required field(s): {joined_names}")


def required_step_scalar_with_fallback(
    step: object,
    names: Sequence[str],
    *,
    env_index: int,
    cast_fn: Any,
    fallback: Any,
    fallback_name: str,
) -> Any:
    try:
        observed = step_scalar(step, names, env_index=env_index, cast_fn=cast_fn)
    except AttributeError:
        if fallback is MISSING:
            joined_names = ", ".join(names)
            raise AttributeError(
                f"step is missing required field(s): {joined_names}; provide {fallback_name} when unavailable"
            ) from None
        return cast_fn(fallback)

    if fallback is MISSING:
        return observed

    expected = cast_fn(fallback)
    if observed != expected:
        raise ValueError(f"{fallback_name} mismatch: step={observed}, provided={expected}")
    return observed


def optional_step_scalar(step: object, names: Sequence[str], *, env_index: int) -> int | bytes | None:
    for name in names:
        if not hasattr(step, name):
            continue
        value = step_value_for_env(step, name=name, env_index=env_index)
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return int(value)
    return None


def reward_perspective_seat(step: object, *, env_index: int, acting_seat: int | None) -> int:
    aliases: list[tuple[str, int]] = []
    if acting_seat is not None:
        aliases.append(("acting_seat", require_seat(acting_seat, name="acting_seat")))

    for name in ("actor", "to_play_seat", "to_play"):
        if not hasattr(step, name):
            continue
        seat = step_scalar(step, (name,), env_index=env_index, cast_fn=int)
        if seat == -1:
            continue
        aliases.append((name, require_seat(seat, name=name)))

    if not aliases:
        raise AttributeError(
            "decisive terminated step must expose acting_seat or a valid actor, to_play_seat, or to_play"
        )

    canonical_name, canonical_seat = aliases[0]
    for name, seat in aliases[1:]:
        if seat != canonical_seat:
            raise ValueError(f"reward perspective seat mismatch: {canonical_name}={canonical_seat}, {name}={seat}")
    return canonical_seat


__all__ = [
    "MISSING",
    "optional_step_scalar",
    "optional_terminal_winner_seat",
    "require_seat",
    "required_step_scalar_with_fallback",
    "reward_perspective_seat",
    "step_scalar",
    "step_value_for_env",
    "winner_seat_from_terminal_step",
]
