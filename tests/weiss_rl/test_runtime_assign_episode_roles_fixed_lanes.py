from __future__ import annotations

from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.batching.counters import collector_counter_template
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_role_assignment_test_support import role_actor, role_assignment_runtime


def test_assign_episode_roles_uses_mirror_for_nondiverse_lane_when_league_disabled() -> None:
    runtime = role_assignment_runtime()
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = False
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True
    actor = role_actor(
        env_count=2,
        opponent_policy_ids=["old0", "old1"],
        diverse_lane=False,
    )
    counters = collector_counter_template()

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
        counters=cast(Any, counters),
    )

    assert actor.opponent_policy_id_by_env.tolist() == [MIRROR_OPPONENT_POLICY_ID, MIRROR_OPPONENT_POLICY_ID]
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 2
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
    assert counters["pfsp_sampled_envs"] == 0
    assert counters["pfsp_mirror_envs"] == 2
    assert counters["pfsp_heuristic_public_envs"] == 0


def test_assign_episode_roles_prioritizes_fixed_anchor_lanes() -> None:
    runtime = role_assignment_runtime()
    runtime_any = cast(Any, runtime)
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True

    def fake_sample(*, count: int, rng) -> tuple[str, ...]:
        del rng
        runtime_any._pfsp_last_sampled_envs = count
        runtime_any._pfsp_last_recent_envs = count
        return tuple(f"recent_{index}" for index in range(count))

    runtime_any._sample_opponent_policy_ids = fake_sample
    actor = role_actor(
        env_count=4,
        opponent_policy_ids=["old0", "old1", "old2", "old3"],
        fixed_policy_ids=["B2 HeuristicPublic", "b1_noleague_baseline", "", ""],
    )

    QueueRuntime._assign_episode_roles(
        runtime, actor, np.asarray([True, True, True, True], dtype=np.bool_), initial=False
    )

    assert actor.opponent_policy_id_by_env.tolist() == [
        "B2 HeuristicPublic",
        "b1_noleague_baseline",
        "recent_0",
        "recent_1",
    ]
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 1
    assert runtime_any._pfsp_last_noleague_baseline_envs == 1
    assert runtime_any._pfsp_last_recent_envs == 2
