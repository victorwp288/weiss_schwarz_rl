"""Compose learner-side reward shaping rules for runtime collectors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from weiss_rl.runtime.components.rewards.reward_shaping_counters import record_reward_penalty
from weiss_rl.runtime.components.rewards.reward_shaping_mulligan import (
    apply_mulligan_select_with_confirm_penalty,
    mulligan_select_with_confirm_penalty_mask_from_ids,
)
from weiss_rl.runtime.components.rewards.reward_shaping_pass import (
    apply_pass_with_nonpass_penalty,
    pass_penalty_ignored_alternative_family_ids,
    pass_with_nonpass_penalty_mask_from_ids,
    pass_with_nonpass_penalty_mask_from_mask,
)
from weiss_rl.runtime.components.rewards.reward_shaping_plan import (
    COLLECTOR_REWARD_SHAPING_PLAN,
    collector_reward_shaping_plan_payload,
)

__all__ = [
    "COLLECTOR_REWARD_SHAPING_PLAN",
    "apply_collector_reward_shaping",
    "apply_mulligan_select_with_confirm_penalty",
    "apply_pass_with_nonpass_penalty",
    "collector_reward_shaping_plan_payload",
    "mulligan_select_with_confirm_penalty_mask_from_ids",
    "pass_penalty_ignored_alternative_family_ids",
    "pass_with_nonpass_penalty_mask_from_ids",
    "pass_with_nonpass_penalty_mask_from_mask",
]


def apply_collector_reward_shaping(
    rewards: np.ndarray,
    actions: np.ndarray,
    *,
    counters: dict[str, int],
    pass_action_id: int,
    pass_with_nonpass_penalty: float,
    mulligan_select_with_confirm_penalty: float,
    action_family_index: Mapping[str, int] | None = None,
    legal_ids: np.ndarray | None = None,
    legal_offsets: np.ndarray | None = None,
    legal_action_meta: np.ndarray | None = None,
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply collector-side reward shaping and update collector counters."""

    pass_rule = COLLECTOR_REWARD_SHAPING_PLAN[0]
    shaped, penalty_count, penalty_total_micros = apply_pass_with_nonpass_penalty(
        rewards,
        actions,
        pass_action_id=int(pass_action_id),
        penalty=float(pass_with_nonpass_penalty),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        ignored_alternative_family_ids=pass_penalty_ignored_alternative_family_ids(action_family_index),
        legal_mask=legal_mask,
    )
    record_reward_penalty(
        counters,
        counter_prefix=pass_rule.rule_id,
        count=penalty_count,
        total_micros=penalty_total_micros,
    )

    if legal_ids is None or legal_offsets is None:
        return shaped

    family_index = {} if action_family_index is None else action_family_index
    mulligan_rule = COLLECTOR_REWARD_SHAPING_PLAN[1]
    shaped, penalty_count, penalty_total_micros = apply_mulligan_select_with_confirm_penalty(
        shaped,
        actions,
        penalty=float(mulligan_select_with_confirm_penalty),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
        mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
    )
    record_reward_penalty(
        counters,
        counter_prefix=mulligan_rule.rule_id,
        count=penalty_count,
        total_micros=penalty_total_micros,
    )
    return shaped
