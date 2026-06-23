"""Actor rollout action counters and learner-side reward shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from weiss_rl.actors.action_selection import ActorActionSelection
from weiss_rl.diagnostics.action_diagnostics import (
    ActionSequenceState,
    make_action_sequence_state,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)
from weiss_rl.runtime.components.reward_shaping import apply_pass_with_nonpass_penalty

ActorLegalityLayout = Literal["i16_legal_ids", "mask"]


@dataclass(slots=True)
class ActorActionAccounting:
    counters: dict[str, int]
    sequence_state: ActionSequenceState


def make_actor_action_accounting(num_envs: int) -> ActorActionAccounting:
    return ActorActionAccounting(
        counters={
            "total_actions": 0,
            "pass_actions": 0,
            "main_move_actions": 0,
            "pass_with_nonpass_available": 0,
            "pass_with_nonpass_penalty_count": 0,
            "pass_with_nonpass_penalty_total_micros": 0,
            "max_consecutive_main_moves": 0,
        },
        sequence_state=make_action_sequence_state(num_envs),
    )


def record_actor_actions(
    *,
    accounting: ActorActionAccounting,
    layout_name: ActorLegalityLayout,
    selection: ActorActionSelection,
    rewards: np.ndarray,
    pass_action_id: int,
    pass_with_nonpass_penalty: float,
    main_move_action: np.ndarray | None = None,
) -> np.ndarray:
    if layout_name == "i16_legal_ids":
        legal_ids, legal_offsets = _require_ids_legality(selection)
        update_action_summary_from_ids(
            counters=accounting.counters,
            state=accounting.sequence_state,
            actions=selection.actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            pass_action_id=pass_action_id,
            main_move_action=main_move_action,
        )
        reward_shaped, penalty_count, penalty_total_micros = apply_pass_with_nonpass_penalty(
            rewards,
            selection.actions,
            pass_action_id=pass_action_id,
            penalty=float(pass_with_nonpass_penalty),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
    else:
        legal_mask = _require_mask_legality(selection)
        update_action_summary_from_mask(
            counters=accounting.counters,
            state=accounting.sequence_state,
            actions=selection.actions,
            legal_mask=legal_mask,
            pass_action_id=pass_action_id,
            main_move_action=main_move_action,
        )
        reward_shaped, penalty_count, penalty_total_micros = apply_pass_with_nonpass_penalty(
            rewards,
            selection.actions,
            pass_action_id=pass_action_id,
            penalty=float(pass_with_nonpass_penalty),
            legal_mask=legal_mask,
        )

    accounting.counters["pass_with_nonpass_penalty_count"] += penalty_count
    accounting.counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
    return reward_shaped


def _require_ids_legality(selection: ActorActionSelection) -> tuple[np.ndarray, np.ndarray]:
    if selection.legal_ids is None or selection.legal_offsets is None:
        raise ValueError("ids-offset action accounting requires legal_ids and legal_offsets")
    return selection.legal_ids, selection.legal_offsets


def _require_mask_legality(selection: ActorActionSelection) -> np.ndarray:
    if selection.legal_mask is None:
        raise ValueError("mask action accounting requires legal_mask")
    return selection.legal_mask


__all__ = [
    "ActorActionAccounting",
    "ActorLegalityLayout",
    "make_actor_action_accounting",
    "record_actor_actions",
]
