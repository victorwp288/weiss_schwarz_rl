from __future__ import annotations

from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.batching.counters import collector_counter_template
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_role_assignment_test_support import role_actor, role_assignment_runtime


def test_assign_episode_roles_records_per_policy_exposure_counters() -> None:
    runtime = role_assignment_runtime()
    runtime_any = cast(Any, runtime)

    def fake_sample(*, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        assert count == 3
        assert rng is actor.rng
        runtime_any._pfsp_last_sampled_envs = 3
        runtime_any._pfsp_last_champion_envs = 2
        runtime_any._pfsp_last_hard_negative_envs = 1
        runtime_any._pfsp_last_sampled_policy_envs = {"champ a": 2, "hard/b": 1}
        runtime_any._pfsp_last_champion_policy_envs = {"champ a": 2}
        runtime_any._pfsp_last_hard_negative_policy_envs = {"hard/b": 1}
        return ("champ a", "hard/b", "champ a")

    actor = role_actor(
        env_count=3,
        opponent_policy_ids=["unknown", "unknown", "unknown"],
        diverse_lane=True,
        rng_seed=3,
    )
    runtime_any._sample_opponent_policy_ids = fake_sample
    counters = collector_counter_template()

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.ones((3,), dtype=np.bool_),
        initial=True,
        counters=counters,
    )

    assert counters["pfsp_champion_envs"] == 2
    assert counters["pfsp_hard_negative_envs"] == 1
    assert counters["pfsp_sampled_policy_envs__champ_a"] == 2
    assert counters["pfsp_sampled_policy_envs__hard_b"] == 1
    assert counters["pfsp_champion_policy_envs__champ_a"] == 2
    assert counters["pfsp_hard_negative_policy_envs__hard_b"] == 1


def test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane() -> None:
    runtime = role_assignment_runtime()
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    weighted_calls: list[int] = []

    def sample_weighted(*, count, rng):
        del rng
        weighted_calls.append(int(count))
        return ("seed_recent_a", "B2 HeuristicPublic")[:count]

    runtime_any._sample_opponent_policy_ids = sample_weighted
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: False
    actor = role_actor(
        env_count=2,
        opponent_policy_ids=[MIRROR_OPPONENT_POLICY_ID, MIRROR_OPPONENT_POLICY_ID],
        diverse_lane=True,
    )

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
    )

    assert actor.opponent_policy_id_by_env.tolist() == ["seed_recent_a", "B2 HeuristicPublic"]
    assert weighted_calls == [2]
