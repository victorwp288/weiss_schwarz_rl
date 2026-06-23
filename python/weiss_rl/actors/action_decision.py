"""Actor policy logits, legal action selection, and numeric fault checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.actors.action_selection import ActorActionSelection, select_legal_ids_actions, select_mask_actions
from weiss_rl.actors.actor_worker_helpers import policy_logits
from weiss_rl.actors.unroll_batch import ActorUnrollBuffers, LayoutName
from weiss_rl.core.masking import MaskingAnomalyCounters


@dataclass(frozen=True)
class ActorActionDecision:
    logits: np.ndarray
    selection: ActorActionSelection

    @property
    def actions(self) -> np.ndarray:
        return self.selection.actions


def choose_actor_actions(
    worker: Any,
    *,
    batch: Any,
    policy_logits_fn: Any,
    obs: np.ndarray,
    to_play: np.ndarray,
    decision_id: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
    buffers: ActorUnrollBuffers,
    t: int,
    num_envs: int,
    action_space: int,
    layout_name: LayoutName,
    rng: np.random.Generator,
    anomaly: MaskingAnomalyCounters,
    pass_action_id: int | None,
) -> ActorActionDecision:
    logits = policy_logits(policy_logits_fn, obs, to_play)
    if logits.shape != (num_envs, action_space):
        raise ValueError(f"policy_logits_fn must return shape (N, A)=({num_envs}, {action_space})")
    if not np.all(np.isfinite(logits)):
        worker._raise_numeric_fault(
            "non-finite actor policy logits",
            step=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
        )

    if layout_name == "i16_legal_ids":
        selection = select_legal_ids_actions(
            batch=batch,
            logits=logits,
            rng=rng,
            counters=anomaly,
            pass_action_id=pass_action_id,
            offset_base=buffers.next_legal_offset,
        )
        assert selection.legal_ids is not None
        assert selection.legal_offsets is not None
        assert selection.unroll_legal_ids is not None
        assert selection.unroll_legal_offsets is not None
        buffers.append_legal_ids(selection.unroll_legal_ids, selection.unroll_legal_offsets)
        _validate_ids_selection(
            worker,
            selection=selection,
            logits=logits,
            t=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
        )
    else:
        selection = select_mask_actions(
            batch=batch,
            logits=logits,
            action_space=action_space,
            rng=rng,
            counters=anomaly,
            pass_action_id=pass_action_id,
        )
        assert selection.legal_mask is not None
        buffers.record_legal_mask(t, selection.legal_mask)
        _validate_mask_selection(
            worker,
            selection=selection,
            logits=logits,
            t=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
        )

    return ActorActionDecision(logits=logits, selection=selection)


def _validate_ids_selection(
    worker: Any,
    *,
    selection: ActorActionSelection,
    logits: np.ndarray,
    t: int,
    obs: np.ndarray,
    to_play: np.ndarray,
    decision_id: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
) -> None:
    if not np.all(np.isfinite(selection.logp)):
        worker._raise_numeric_fault(
            "non-finite actor sampled logp",
            step=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
            actions=selection.actions,
            logp=selection.logp,
            entropy=selection.entropy,
            legal_ids=selection.legal_ids,
            legal_offsets=selection.legal_offsets,
        )
    if not np.all(np.isfinite(selection.entropy)):
        worker._raise_numeric_fault(
            "non-finite actor sampled entropy",
            step=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
            actions=selection.actions,
            logp=selection.logp,
            entropy=selection.entropy,
            legal_ids=selection.legal_ids,
            legal_offsets=selection.legal_offsets,
        )


def _validate_mask_selection(
    worker: Any,
    *,
    selection: ActorActionSelection,
    logits: np.ndarray,
    t: int,
    obs: np.ndarray,
    to_play: np.ndarray,
    decision_id: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
) -> None:
    if not np.all(np.isfinite(selection.logp)):
        worker._raise_numeric_fault(
            "non-finite actor sampled logp",
            step=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
            actions=selection.actions,
            logp=selection.logp,
            entropy=selection.entropy,
            legal_mask=selection.legal_mask,
        )
    if not np.all(np.isfinite(selection.entropy)):
        worker._raise_numeric_fault(
            "non-finite actor sampled entropy",
            step=t,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
            actions=selection.actions,
            logp=selection.logp,
            entropy=selection.entropy,
            legal_mask=selection.legal_mask,
        )


__all__ = ["ActorActionDecision", "choose_actor_actions"]
