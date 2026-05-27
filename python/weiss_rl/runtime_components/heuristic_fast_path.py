"""Capability checks for heuristic-public runtime fast paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def simulator_native_fixed_opponent_available(
    actor: Any | None,
    *,
    fixed_opponent_backend: str,
) -> bool:
    if actor is None:
        return False
    if str(fixed_opponent_backend) != "simulator_native":
        return False
    pool = getattr(getattr(actor, "env", None), "pool", None)
    return callable(getattr(pool, "choose_heuristic_public_actions_into", None))


def actor_fixed_opponents_all_heuristic_public(
    actor: Any,
    *,
    fixed_opponent_policy_is_active: Callable[[str], bool],
    heuristic_policy_id: str,
) -> bool:
    fixed_policy_ids = getattr(actor, "fixed_opponent_policy_id_by_env", None)
    if fixed_policy_ids is None:
        return True
    fixed_policy_ids = np.asarray(fixed_policy_ids, dtype=object)
    for policy_id in fixed_policy_ids.tolist():
        policy_name = str(policy_id).strip()
        if not policy_name:
            continue
        if not fixed_opponent_policy_is_active(policy_name):
            continue
        if policy_name != heuristic_policy_id:
            return False
    return True


def can_collect_all_heuristic_ids_fast(
    actor: Any,
    *,
    actor_policy_backend: str,
    active_actor_heuristic_fraction: float,
    fixed_opponent_backend: str,
    teacher_policy: Any | None,
    league_config: Any | None,
    active_heuristic_public_mix_fraction: float,
    fixed_opponent_policy_is_active: Callable[[str], bool],
    heuristic_policy_id: str,
) -> bool:
    if str(getattr(actor, "layout_name", "")) != "i16_legal_ids":
        return False
    if str(actor_policy_backend) != "heuristic_public":
        return False
    if bool(getattr(actor, "force_model_policy_lane", False)):
        return False
    if float(active_actor_heuristic_fraction) < 1.0:
        return False
    if not simulator_native_fixed_opponent_available(
        actor,
        fixed_opponent_backend=fixed_opponent_backend,
    ):
        return False
    if teacher_policy is None:
        return False
    if league_config is None:
        return False
    if float(active_heuristic_public_mix_fraction) < 1.0:
        return False
    if not actor_fixed_opponents_all_heuristic_public(
        actor,
        fixed_opponent_policy_is_active=fixed_opponent_policy_is_active,
        heuristic_policy_id=heuristic_policy_id,
    ):
        return False
    opponent_policy_ids = np.asarray(getattr(actor, "opponent_policy_id_by_env", ()), dtype=object)
    if opponent_policy_ids.size == 0:
        return False
    return all(str(policy_id) == heuristic_policy_id for policy_id in opponent_policy_ids.tolist())


def can_collect_all_heuristic_ids_native_rollout(
    actor: Any,
    *,
    heuristic_native_rollout_enabled: bool,
    actor_policy_backend: str,
    active_actor_heuristic_fraction: float,
    fixed_opponent_backend: str,
    teacher_policy: Any | None,
    league_config: Any | None,
    active_heuristic_public_mix_fraction: float,
    fixed_opponent_policy_is_active: Callable[[str], bool],
    heuristic_policy_id: str,
    actor_behavior_values_required: bool,
    should_track_heuristic_actor_hidden_state: bool,
) -> bool:
    if not bool(heuristic_native_rollout_enabled):
        return False
    if not can_collect_all_heuristic_ids_fast(
        actor,
        actor_policy_backend=actor_policy_backend,
        active_actor_heuristic_fraction=active_actor_heuristic_fraction,
        fixed_opponent_backend=fixed_opponent_backend,
        teacher_policy=teacher_policy,
        league_config=league_config,
        active_heuristic_public_mix_fraction=active_heuristic_public_mix_fraction,
        fixed_opponent_policy_is_active=fixed_opponent_policy_is_active,
        heuristic_policy_id=heuristic_policy_id,
    ):
        return False
    if bool(actor_behavior_values_required):
        return False
    if bool(should_track_heuristic_actor_hidden_state):
        return False
    pool = getattr(getattr(actor, "env", None), "pool", None)
    rollout_into = getattr(pool, "rollout_heuristic_public_into_i16_legal_ids", None)
    reset_done_into = getattr(pool, "reset_done_into_i16_legal_ids", None)
    return callable(rollout_into) and callable(reset_done_into)
