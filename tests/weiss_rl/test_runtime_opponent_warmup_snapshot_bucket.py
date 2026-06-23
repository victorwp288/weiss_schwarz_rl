from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime import QueueRuntime

from .runtime_opponent_sampling_test_support import make_sampling_runtime


def test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            warmup=SimpleNamespace(first_updates=200000),
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.0,
                noleague_baseline_mix_fraction=0.0,
                noleague_baseline_mix_end_updates=-1,
                warmup_snapshot_mix_fraction=1.0,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_candidate_ids=("seed_recent_a", "seed_recent_b"),
        opponent_models={"seed_recent_a": object(), "seed_recent_b": object()},
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=4, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert set(sampled) <= {"seed_recent_a", "seed_recent_b"}
    assert len(sampled) == 4
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_warmup_snapshot_envs == 4


def test_sample_opponent_policy_ids_respects_fractional_warmup_snapshot_mix_before_pfsp_ready() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            warmup=SimpleNamespace(first_updates=200000),
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.25,
                noleague_baseline_mix_fraction=0.0,
                noleague_baseline_mix_end_updates=-1,
                warmup_snapshot_mix_fraction=0.5,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=0.0,
            ),
        ),
        opponent_candidate_ids=("seed_recent_a", "seed_recent_b"),
        opponent_heuristic_policies={HEURISTIC_PUBLIC_POLICY_ID: object()},
        opponent_models={"seed_recent_a": object(), "seed_recent_b": object()},
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=200, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert len(sampled) == 200
    assert runtime_any._pfsp_last_heuristic_public_envs > 0
    assert runtime_any._pfsp_last_warmup_snapshot_envs > 0
    assert runtime_any._pfsp_last_mirror_envs > 0
    assert runtime_any._pfsp_last_sampled_envs == (
        runtime_any._pfsp_last_heuristic_public_envs + runtime_any._pfsp_last_warmup_snapshot_envs
    )
