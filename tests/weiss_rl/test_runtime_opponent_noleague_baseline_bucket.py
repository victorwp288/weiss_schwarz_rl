from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_sampling_test_support import make_sampling_runtime


def test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.0,
                noleague_baseline_mix_fraction=1.0,
                noleague_baseline_mix_end_updates=-1,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_models={NOLEAGUE_BASELINE_POLICY_ID: object()},
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=4, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert sampled == (
        NOLEAGUE_BASELINE_POLICY_ID,
        NOLEAGUE_BASELINE_POLICY_ID,
        NOLEAGUE_BASELINE_POLICY_ID,
        NOLEAGUE_BASELINE_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_noleague_baseline_envs == 4


def test_sample_opponent_policy_ids_disables_noleague_mix_after_end_update() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.0,
                noleague_baseline_mix_fraction=1.0,
                noleague_baseline_mix_end_updates=1,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_models={NOLEAGUE_BASELINE_POLICY_ID: object()},
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
    assert runtime_any._pfsp_last_noleague_baseline_envs == 0
