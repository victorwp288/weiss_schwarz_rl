from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime


def role_assignment_runtime() -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    reset_role_assignment_counters(runtime)
    return runtime


def reset_role_assignment_counters(runtime: QueueRuntime) -> None:
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._pfsp_last_sampled_policy_envs = {}
    runtime_any._pfsp_last_heuristic_public_policy_envs = {}
    runtime_any._pfsp_last_heuristic_public_variant_policy_envs = {}
    runtime_any._pfsp_last_noleague_baseline_policy_envs = {}
    runtime_any._pfsp_last_champion_policy_envs = {}
    runtime_any._pfsp_last_recent_policy_envs = {}
    runtime_any._pfsp_last_hard_negative_policy_envs = {}
    runtime_any._pfsp_last_warmup_snapshot_policy_envs = {}


def role_actor(
    *,
    env_count: int,
    opponent_policy_ids: list[str] | None = None,
    fixed_policy_ids: list[str] | None = None,
    diverse_lane: bool = True,
    rng_seed: int = 7,
) -> Any:
    return SimpleNamespace(
        actor_id=0,
        rng=np.random.default_rng(rng_seed),
        focal_seat_by_env=np.asarray([index % 2 for index in range(env_count)], dtype=np.int64),
        opponent_policy_id_by_env=np.asarray(
            opponent_policy_ids if opponent_policy_ids is not None else [f"old{index}" for index in range(env_count)],
            dtype=object,
        ),
        fixed_opponent_policy_id_by_env=(
            None if fixed_policy_ids is None else np.asarray(fixed_policy_ids, dtype=object)
        ),
        diverse_opponent_lane=bool(diverse_lane),
    )
