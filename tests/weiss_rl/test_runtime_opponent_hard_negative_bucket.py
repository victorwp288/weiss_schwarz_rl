from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime

from .runtime_opponent_sampling_test_support import make_sampling_runtime


def test_sample_opponent_policy_ids_can_force_hard_negative_bucket() -> None:
    runtime = make_sampling_runtime(
        league_config=SimpleNamespace(
            pfsp_power=2.0,
            pfsp_epsilon_uniform=0.0,
            sampling=SimpleNamespace(
                heuristic_public_start_updates=0,
                heuristic_public_mix_fraction=0.0,
                champion_mix_fraction=0.0,
                hard_negative_mix_fraction=1.0,
            ),
        ),
        opponent_candidate_ids=("policy_hard", "policy_recent"),
        opponent_hard_negative_ids=("policy_hard",),
        opponent_recent_ids=("policy_recent",),
        opponent_models={"policy_hard": object(), "policy_recent": object()},
        pfsp_ready=True,
    )

    sampled = QueueRuntime._sample_opponent_policy_ids(runtime, count=4, rng=np.random.default_rng(7))
    runtime_any = cast(Any, runtime)

    assert sampled == ("policy_hard", "policy_hard", "policy_hard", "policy_hard")
    assert runtime_any._pfsp_last_hard_negative_envs == 4
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
