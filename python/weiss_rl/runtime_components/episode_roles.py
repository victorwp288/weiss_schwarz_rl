"""Episode role and opponent assignment for queue runtime actors."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.runtime_components.actor_state import _ActorState
from weiss_rl.runtime_components.policy_ids import MIRROR_OPPONENT_POLICY_ID

_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID


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
        fixed_heuristic_public_count = 0
        fixed_noleague_baseline_count = 0
        if fixed_policy_ids is not None:
            fixed_policy_ids = np.asarray(fixed_policy_ids, dtype=object)
            fixed_assign_mask = np.asarray(
                [
                    bool(done_flag)
                    and bool(str(policy_id).strip())
                    and self._fixed_opponent_policy_is_active(str(policy_id))
                    for done_flag, policy_id in zip(done_array.tolist(), fixed_policy_ids.tolist(), strict=True)
                ],
                dtype=np.bool_,
            )
            if np.any(fixed_assign_mask):
                actor.opponent_policy_id_by_env[fixed_assign_mask] = fixed_policy_ids[fixed_assign_mask]
                remaining_mask = remaining_mask & ~fixed_assign_mask
                fixed_heuristic_public_count = int(
                    np.count_nonzero(fixed_policy_ids[fixed_assign_mask] == HEURISTIC_PUBLIC_POLICY_ID)
                )
                fixed_noleague_baseline_count = int(
                    np.count_nonzero(fixed_policy_ids[fixed_assign_mask] == _NOLEAGUE_BASELINE_POLICY_ID)
                )

        remaining_count = int(np.count_nonzero(remaining_mask))
        if bool(getattr(actor, "diverse_opponent_lane", True)):
            sampled_policy_ids = self._sample_opponent_policy_ids(count=remaining_count, rng=actor.rng)
            actor.opponent_policy_id_by_env[remaining_mask] = np.asarray(sampled_policy_ids, dtype=object)
        else:
            use_heuristic_anchor_lane = bool(
                getattr(self, "_league_enabled", False)
            ) and self._fixed_opponent_policy_is_active(HEURISTIC_PUBLIC_POLICY_ID)
            if remaining_count > 0:
                actor.opponent_policy_id_by_env[remaining_mask] = (
                    HEURISTIC_PUBLIC_POLICY_ID if use_heuristic_anchor_lane else MIRROR_OPPONENT_POLICY_ID
                )
            self._pfsp_last_sampled_envs = remaining_count if use_heuristic_anchor_lane else 0
            self._pfsp_last_mirror_envs = 0 if use_heuristic_anchor_lane else remaining_count
            self._pfsp_last_heuristic_public_envs = remaining_count if use_heuristic_anchor_lane else 0
            self._pfsp_last_heuristic_public_variant_envs = 0
            self._pfsp_last_noleague_baseline_envs = 0
            self._pfsp_last_champion_envs = 0
            self._pfsp_last_recent_envs = 0
            self._pfsp_last_hard_negative_envs = 0
            self._pfsp_last_warmup_snapshot_envs = 0
            heuristic_anchor_counts = (
                {HEURISTIC_PUBLIC_POLICY_ID: remaining_count}
                if use_heuristic_anchor_lane and remaining_count > 0
                else {}
            )
            self._pfsp_last_sampled_policy_envs = dict(heuristic_anchor_counts)
            self._pfsp_last_heuristic_public_policy_envs = dict(heuristic_anchor_counts)
            self._pfsp_last_heuristic_public_variant_policy_envs: dict[str, int] = {}
            self._pfsp_last_noleague_baseline_policy_envs: dict[str, int] = {}
            self._pfsp_last_champion_policy_envs: dict[str, int] = {}
            self._pfsp_last_recent_policy_envs: dict[str, int] = {}
            self._pfsp_last_hard_negative_policy_envs: dict[str, int] = {}
            self._pfsp_last_warmup_snapshot_policy_envs: dict[str, int] = {}
        if fixed_heuristic_public_count or fixed_noleague_baseline_count:
            self._pfsp_last_sampled_envs += fixed_heuristic_public_count + fixed_noleague_baseline_count
            self._pfsp_last_heuristic_public_envs += fixed_heuristic_public_count
            self._pfsp_last_noleague_baseline_envs += fixed_noleague_baseline_count
        if counters is not None:
            counters["pfsp_sampled_envs"] += int(self._pfsp_last_sampled_envs)
            counters["pfsp_mirror_envs"] += int(self._pfsp_last_mirror_envs)
            counters["pfsp_heuristic_public_envs"] += int(self._pfsp_last_heuristic_public_envs)
            counters["pfsp_heuristic_public_variant_envs"] += int(
                getattr(self, "_pfsp_last_heuristic_public_variant_envs", 0)
            )
            counters["pfsp_noleague_baseline_envs"] += int(self._pfsp_last_noleague_baseline_envs)
            counters["pfsp_champion_envs"] += int(self._pfsp_last_champion_envs)
            counters["pfsp_recent_envs"] += int(self._pfsp_last_recent_envs)
            counters["pfsp_hard_negative_envs"] += int(self._pfsp_last_hard_negative_envs)
            counters["pfsp_warmup_snapshot_envs"] += int(getattr(self, "_pfsp_last_warmup_snapshot_envs", 0))
            _add_policy_exposure_counters(
                counters,
                group_name="sampled",
                policy_counts=getattr(self, "_pfsp_last_sampled_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="heuristic_public",
                policy_counts=getattr(self, "_pfsp_last_heuristic_public_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="heuristic_public_variant",
                policy_counts=getattr(self, "_pfsp_last_heuristic_public_variant_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="noleague_baseline",
                policy_counts=getattr(self, "_pfsp_last_noleague_baseline_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="champion",
                policy_counts=getattr(self, "_pfsp_last_champion_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="recent",
                policy_counts=getattr(self, "_pfsp_last_recent_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="hard_negative",
                policy_counts=getattr(self, "_pfsp_last_hard_negative_policy_envs", {}),
            )
            _add_policy_exposure_counters(
                counters,
                group_name="warmup_snapshot",
                policy_counts=getattr(self, "_pfsp_last_warmup_snapshot_policy_envs", {}),
            )
            if fixed_heuristic_public_count > 0:
                fixed_counts = {HEURISTIC_PUBLIC_POLICY_ID: fixed_heuristic_public_count}
                _add_policy_exposure_counters(counters, group_name="sampled", policy_counts=fixed_counts)
                _add_policy_exposure_counters(counters, group_name="heuristic_public", policy_counts=fixed_counts)
            if fixed_noleague_baseline_count > 0:
                fixed_counts = {_NOLEAGUE_BASELINE_POLICY_ID: fixed_noleague_baseline_count}
                _add_policy_exposure_counters(counters, group_name="sampled", policy_counts=fixed_counts)
                _add_policy_exposure_counters(counters, group_name="noleague_baseline", policy_counts=fixed_counts)
