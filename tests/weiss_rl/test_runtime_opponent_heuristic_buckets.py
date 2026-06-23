from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
)
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_sampling_test_support import make_sampling_runtime


def test_sample_opponent_policy_ids_can_force_heuristic_public_bucket_before_pfsp_ready() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=1.0,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_heuristic_policies={"B2 HeuristicPublic": object()},
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=4, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert sampled == (
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 4


def test_sample_opponent_policy_ids_respects_heuristic_public_mix_anneal_end_update() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=1.0,
                heuristic_public_mix_end_updates=1,
                heuristic_public_final_mix_fraction=0.0,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_heuristic_policies={"B2 HeuristicPublic": object()},
        reference_update=1,
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=4, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert sampled == (
        MIRROR_OPPONENT_POLICY_ID,
        MIRROR_OPPONENT_POLICY_ID,
        MIRROR_OPPONENT_POLICY_ID,
        MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_sample_opponent_policy_ids_can_force_heuristic_public_variant_bucket_before_pfsp_ready() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.0,
                heuristic_public_variant_mix_fraction=1.0,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
            pfsp_power=1.5,
            pfsp_epsilon_uniform=0.3,
        ),
        opponent_heuristic_policies={
            HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
            HEURISTIC_PUBLIC_CONTROL_POLICY_ID: object(),
        },
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=8, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert set(sampled).issubset({HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID})
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_variant_envs == 8
