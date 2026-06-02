"""Episode role and opponent assignment for queue runtime actors."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.runtime.components.actor_state import _ActorState
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID

_PFSP_ENV_COUNTER_FIELDS = (
    ("pfsp_sampled_envs", "_pfsp_last_sampled_envs"),
    ("pfsp_mirror_envs", "_pfsp_last_mirror_envs"),
    ("pfsp_heuristic_public_envs", "_pfsp_last_heuristic_public_envs"),
    ("pfsp_heuristic_public_variant_envs", "_pfsp_last_heuristic_public_variant_envs"),
    ("pfsp_noleague_baseline_envs", "_pfsp_last_noleague_baseline_envs"),
    ("pfsp_champion_envs", "_pfsp_last_champion_envs"),
    ("pfsp_recent_envs", "_pfsp_last_recent_envs"),
    ("pfsp_hard_negative_envs", "_pfsp_last_hard_negative_envs"),
    ("pfsp_warmup_snapshot_envs", "_pfsp_last_warmup_snapshot_envs"),
)

_PFSP_POLICY_COUNTER_FIELDS = (
    ("sampled", "_pfsp_last_sampled_policy_envs"),
    ("heuristic_public", "_pfsp_last_heuristic_public_policy_envs"),
    ("heuristic_public_variant", "_pfsp_last_heuristic_public_variant_policy_envs"),
    ("noleague_baseline", "_pfsp_last_noleague_baseline_policy_envs"),
    ("champion", "_pfsp_last_champion_policy_envs"),
    ("recent", "_pfsp_last_recent_policy_envs"),
    ("hard_negative", "_pfsp_last_hard_negative_policy_envs"),
    ("warmup_snapshot", "_pfsp_last_warmup_snapshot_policy_envs"),
)


@dataclass(frozen=True, slots=True)
class FixedOpponentRoleAssignment:
    assign_mask: np.ndarray
    remaining_mask: np.ndarray
    heuristic_public_envs: int = 0
    noleague_baseline_envs: int = 0


@dataclass(frozen=True, slots=True)
class NonDiverseOpponentRoleAssignment:
    policy_id: str
    sampled_envs: int
    mirror_envs: int
    heuristic_public_envs: int
    sampled_policy_envs: Mapping[str, int]
    heuristic_public_policy_envs: Mapping[str, int]


def _metric_safe_policy_id(policy_id: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(policy_id).strip()).strip("_").lower()
    return sanitized or "unknown"


def _add_policy_exposure_counters(
    counters: dict[str, int],
    *,
    group_name: str,
    policy_counts: Any,
) -> None:
    for policy_id, count in dict(policy_counts or {}).items():
        env_count = int(count)
        if env_count <= 0:
            continue
        key = f"pfsp_{group_name}_policy_envs__{_metric_safe_policy_id(str(policy_id))}"
        counters[key] = int(counters.get(key, 0)) + env_count


def resolve_fixed_opponent_role_assignment(
    *,
    done: np.ndarray,
    fixed_policy_ids: np.ndarray | None,
    policy_is_active: Callable[[str], bool],
) -> FixedOpponentRoleAssignment:
    done_array = np.asarray(done, dtype=np.bool_)
    if fixed_policy_ids is None:
        empty_mask = np.zeros(done_array.shape, dtype=np.bool_)
        return FixedOpponentRoleAssignment(assign_mask=empty_mask, remaining_mask=done_array.copy())
    fixed_ids = np.asarray(fixed_policy_ids, dtype=object)
    if fixed_ids.shape != done_array.shape:
        raise ValueError(f"fixed_policy_ids must have shape {done_array.shape}, got {fixed_ids.shape}")
    assign_mask = np.asarray(
        [
            bool(done_flag) and bool(str(policy_id).strip()) and policy_is_active(str(policy_id))
            for done_flag, policy_id in zip(done_array.tolist(), fixed_ids.tolist(), strict=True)
        ],
        dtype=np.bool_,
    )
    return FixedOpponentRoleAssignment(
        assign_mask=assign_mask,
        remaining_mask=done_array & ~assign_mask,
        heuristic_public_envs=int(np.count_nonzero(fixed_ids[assign_mask] == HEURISTIC_PUBLIC_POLICY_ID)),
        noleague_baseline_envs=int(np.count_nonzero(fixed_ids[assign_mask] == _NOLEAGUE_BASELINE_POLICY_ID)),
    )


def nondiverse_opponent_role_assignment(
    *,
    remaining_count: int,
    league_enabled: bool,
    heuristic_anchor_active: bool,
) -> NonDiverseOpponentRoleAssignment:
    use_heuristic_anchor_lane = bool(league_enabled) and bool(heuristic_anchor_active)
    policy_id = HEURISTIC_PUBLIC_POLICY_ID if use_heuristic_anchor_lane else MIRROR_OPPONENT_POLICY_ID
    heuristic_anchor_counts = (
        {HEURISTIC_PUBLIC_POLICY_ID: int(remaining_count)}
        if use_heuristic_anchor_lane and int(remaining_count) > 0
        else {}
    )
    return NonDiverseOpponentRoleAssignment(
        policy_id=policy_id,
        sampled_envs=int(remaining_count) if use_heuristic_anchor_lane else 0,
        mirror_envs=0 if use_heuristic_anchor_lane else int(remaining_count),
        heuristic_public_envs=int(remaining_count) if use_heuristic_anchor_lane else 0,
        sampled_policy_envs=heuristic_anchor_counts,
        heuristic_public_policy_envs=heuristic_anchor_counts,
    )


def add_fixed_anchor_exposure_to_last_sample(
    runtime: Any,
    *,
    heuristic_public_envs: int,
    noleague_baseline_envs: int,
) -> None:
    if not heuristic_public_envs and not noleague_baseline_envs:
        return
    runtime._pfsp_last_sampled_envs += int(heuristic_public_envs) + int(noleague_baseline_envs)
    runtime._pfsp_last_heuristic_public_envs += int(heuristic_public_envs)
    runtime._pfsp_last_noleague_baseline_envs += int(noleague_baseline_envs)


def accumulate_last_pfsp_exposure_counters(
    counters: dict[str, int],
    runtime: Any,
    *,
    fixed_heuristic_public_envs: int = 0,
    fixed_noleague_baseline_envs: int = 0,
) -> None:
    for counter_key, runtime_attr in _PFSP_ENV_COUNTER_FIELDS:
        counters[counter_key] += int(getattr(runtime, runtime_attr, 0))
    for group_name, runtime_attr in _PFSP_POLICY_COUNTER_FIELDS:
        _add_policy_exposure_counters(
            counters,
            group_name=group_name,
            policy_counts=getattr(runtime, runtime_attr, {}),
        )
    if fixed_heuristic_public_envs > 0:
        fixed_counts = {HEURISTIC_PUBLIC_POLICY_ID: int(fixed_heuristic_public_envs)}
        _add_policy_exposure_counters(counters, group_name="sampled", policy_counts=fixed_counts)
        _add_policy_exposure_counters(counters, group_name="heuristic_public", policy_counts=fixed_counts)
    if fixed_noleague_baseline_envs > 0:
        fixed_counts = {_NOLEAGUE_BASELINE_POLICY_ID: int(fixed_noleague_baseline_envs)}
        _add_policy_exposure_counters(counters, group_name="sampled", policy_counts=fixed_counts)
        _add_policy_exposure_counters(counters, group_name="noleague_baseline", policy_counts=fixed_counts)


class QueueRuntimeEpisodeRolesMixin:
    def _assign_episode_roles(
        self: Any,
        actor: _ActorState,
        done: np.ndarray,
        *,
        initial: bool = False,
        counters: dict[str, int] | None = None,
    ) -> None:
        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.shape != actor.focal_seat_by_env.shape:
            raise ValueError(f"done must have shape {actor.focal_seat_by_env.shape}, got {done_array.shape}")
        if not np.any(done_array):
            return
        if initial:
            actor.focal_seat_by_env[done_array] = (actor.actor_id + np.flatnonzero(done_array)) % 2
        else:
            actor.focal_seat_by_env[done_array] = 1 - actor.focal_seat_by_env[done_array]

        remaining_mask = done_array.copy()
        fixed_policy_ids = getattr(actor, "fixed_opponent_policy_id_by_env", None)
        fixed_assignment = resolve_fixed_opponent_role_assignment(
            done=done_array,
            fixed_policy_ids=None if fixed_policy_ids is None else np.asarray(fixed_policy_ids, dtype=object),
            policy_is_active=self._fixed_opponent_policy_is_active,
        )
        if np.any(fixed_assignment.assign_mask):
            assert fixed_policy_ids is not None
            fixed_policy_ids = np.asarray(fixed_policy_ids, dtype=object)
            actor.opponent_policy_id_by_env[fixed_assignment.assign_mask] = fixed_policy_ids[
                fixed_assignment.assign_mask
            ]
            remaining_mask = fixed_assignment.remaining_mask

        remaining_count = int(np.count_nonzero(remaining_mask))
        if bool(getattr(actor, "diverse_opponent_lane", True)):
            sampled_policy_ids = self._sample_opponent_policy_ids(count=remaining_count, rng=actor.rng)
            actor.opponent_policy_id_by_env[remaining_mask] = np.asarray(sampled_policy_ids, dtype=object)
        else:
            nondiverse_assignment = nondiverse_opponent_role_assignment(
                remaining_count=remaining_count,
                league_enabled=bool(getattr(self, "_league_enabled", False)),
                heuristic_anchor_active=self._fixed_opponent_policy_is_active(HEURISTIC_PUBLIC_POLICY_ID),
            )
            if remaining_count > 0:
                actor.opponent_policy_id_by_env[remaining_mask] = nondiverse_assignment.policy_id
            self._pfsp_last_sampled_envs = nondiverse_assignment.sampled_envs
            self._pfsp_last_mirror_envs = nondiverse_assignment.mirror_envs
            self._pfsp_last_heuristic_public_envs = nondiverse_assignment.heuristic_public_envs
            self._pfsp_last_heuristic_public_variant_envs = 0
            self._pfsp_last_noleague_baseline_envs = 0
            self._pfsp_last_champion_envs = 0
            self._pfsp_last_recent_envs = 0
            self._pfsp_last_hard_negative_envs = 0
            self._pfsp_last_warmup_snapshot_envs = 0
            self._pfsp_last_sampled_policy_envs = dict(nondiverse_assignment.sampled_policy_envs)
            self._pfsp_last_heuristic_public_policy_envs = dict(nondiverse_assignment.heuristic_public_policy_envs)
            self._pfsp_last_heuristic_public_variant_policy_envs = dict[str, int]()
            self._pfsp_last_noleague_baseline_policy_envs = dict[str, int]()
            self._pfsp_last_champion_policy_envs = dict[str, int]()
            self._pfsp_last_recent_policy_envs = dict[str, int]()
            self._pfsp_last_hard_negative_policy_envs = dict[str, int]()
            self._pfsp_last_warmup_snapshot_policy_envs = dict[str, int]()
        add_fixed_anchor_exposure_to_last_sample(
            self,
            heuristic_public_envs=fixed_assignment.heuristic_public_envs,
            noleague_baseline_envs=fixed_assignment.noleague_baseline_envs,
        )
        if counters is not None:
            accumulate_last_pfsp_exposure_counters(
                counters,
                self,
                fixed_heuristic_public_envs=fixed_assignment.heuristic_public_envs,
                fixed_noleague_baseline_envs=fixed_assignment.noleague_baseline_envs,
            )
