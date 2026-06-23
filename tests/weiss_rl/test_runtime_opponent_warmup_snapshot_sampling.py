from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime.components.opponents import sample_warmup_snapshot_policy_ids

from .runtime_opponent_sampling_test_support import OpponentSamplingOutcomes


def test_sample_warmup_snapshot_policy_ids_preserves_pfsp_counters() -> None:
    league_config = SimpleNamespace(pfsp_power=2.0, pfsp_epsilon_uniform=0.0)

    result = sample_warmup_snapshot_policy_ids(
        count=6,
        rng=np.random.default_rng(3),
        opponent_candidate_ids=("warm_a", "warm_b"),
        league_config=league_config,
        outcomes=OpponentSamplingOutcomes(),
    )

    assert result.policy_ids == ("warm_a", "warm_a", "warm_a", "warm_a", "warm_a", "warm_a")
    assert result.sampled_envs == 6
    assert result.warmup_snapshot_envs == 6
    assert result.mirror_envs == 0
    assert dict(result.sampled_policy_envs) == {"warm_a": 6}
    assert dict(result.warmup_snapshot_policy_envs) == {"warm_a": 6}
