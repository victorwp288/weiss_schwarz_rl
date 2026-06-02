from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from weiss_rl.core.masking import (
    logits_for_sampling_temperature,
    masked_logp_from_mask,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)
from weiss_rl.diagnostics.action_diagnostics import (
    ActionSequenceState,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)


class CollectorActionStep(NamedTuple):
    next_batch: Any
    actions: np.ndarray
    logp: np.ndarray


def step_env_with_actions(
    *,
    env: Any,
    actions: np.ndarray,
    counters: dict[str, int],
    before_step: Callable[[], None] | None = None,
) -> Any:
    env_started = time.perf_counter()
    if before_step is not None:
        before_step()
    next_batch = env.step(np.asarray(actions).astype(np.uint32, copy=False))
    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
    return next_batch


def fused_step_packed_from_logits_with_logp(
    *,
    env: Any,
    logits: np.ndarray,
    rng: np.random.Generator,
    counters: dict[str, int],
    temperature: float,
) -> CollectorActionStep:
    sample_seeds = rng.integers(0, np.iinfo(np.int64).max, size=int(logits.shape[0]), dtype=np.int64)
    sampling_logits = logits_for_sampling_temperature(logits, temperature=float(temperature))
    env_started = time.perf_counter()
    next_batch, fused_actions, fused_logp = env.step_sample_from_logits_with_logp(sampling_logits, sample_seeds)
    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
    return CollectorActionStep(
        next_batch=next_batch,
        actions=np.asarray(fused_actions, dtype=np.int64),
        logp=np.asarray(fused_logp, dtype=np.float32),
    )


def fused_step_mask_from_logits(
    *,
    env: Any,
    logits: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
    counters: dict[str, int],
    pass_action_id: int,
    temperature: float,
) -> CollectorActionStep:
    sample_seeds = rng.integers(0, np.iinfo(np.int64).max, size=int(logits.shape[0]), dtype=np.int64)
    sampling_logits = logits_for_sampling_temperature(logits, temperature=float(temperature))
    env_started = time.perf_counter()
    next_batch, fused_actions = env.step_sample_from_logits(sampling_logits, sample_seeds)
    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
    actions = np.asarray(fused_actions, dtype=np.int64)
    logp = masked_logp_from_mask(
        sampling_logits,
        legal_mask,
        actions.astype(np.uint32, copy=False),
        pass_action_id=pass_action_id,
    )
    return CollectorActionStep(next_batch=next_batch, actions=actions, logp=logp)


def sample_and_step_packed_from_logits(
    *,
    env: Any,
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    rng: np.random.Generator,
    counters: dict[str, int],
    pass_action_id: int,
    temperature: float,
) -> CollectorActionStep:
    env_started = time.perf_counter()
    actions, logp, _entropy = sample_actions_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        rng=rng,
        pass_action_id=pass_action_id,
        temperature=float(temperature),
    )
    next_batch = env.step(np.asarray(actions, dtype=np.uint32))
    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
    return CollectorActionStep(next_batch=next_batch, actions=actions, logp=logp)


def sample_and_step_mask_from_logits(
    *,
    env: Any,
    logits: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
    counters: dict[str, int],
    pass_action_id: int,
    temperature: float,
) -> CollectorActionStep:
    env_started = time.perf_counter()
    actions, logp, _entropy = sample_actions_from_mask(
        logits,
        legal_mask,
        rng=rng,
        pass_action_id=pass_action_id,
        temperature=float(temperature),
    )
    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
    next_batch = step_env_with_actions(env=env, actions=actions, counters=counters)
    return CollectorActionStep(next_batch=next_batch, actions=actions, logp=logp)


def record_packed_action_summary(
    *,
    counters: dict[str, int],
    state: ActionSequenceState,
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    pass_action_id: int,
    next_batch: Any,
) -> None:
    summary_started = time.perf_counter()
    update_action_summary_from_ids(
        counters=counters,
        state=state,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        pass_action_id=pass_action_id,
        main_move_action=getattr(next_batch, "main_move_action", None),
    )
    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)


def record_mask_action_summary(
    *,
    counters: dict[str, int],
    state: ActionSequenceState,
    actions: np.ndarray,
    legal_mask: np.ndarray,
    pass_action_id: int,
    next_batch: Any,
) -> None:
    summary_started = time.perf_counter()
    update_action_summary_from_mask(
        counters=counters,
        state=state,
        actions=actions,
        legal_mask=legal_mask,
        pass_action_id=pass_action_id,
        main_move_action=getattr(next_batch, "main_move_action", None),
    )
    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)
