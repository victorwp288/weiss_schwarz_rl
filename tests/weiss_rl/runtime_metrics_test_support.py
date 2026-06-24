from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from weiss_rl.runtime.components.batching.metrics import build_runtime_metrics


def _runtime_unroll(
    *,
    t: int,
    n: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((t, n, 1), dtype=np.float32),
        behavior_policy_version=behavior_policy_version,
        counters=counters,
    )


def _build_runtime_metrics_with_defaults(**overrides: Any) -> tuple[dict[str, float], int]:
    kwargs: dict[str, Any] = {
        "selected": [],
        "occupancy_samples": [],
        "now": 2.0,
        "runtime_start": 1.0,
        "runtime_last_metrics_time": 1.0,
        "runtime_cumulative_env_steps": 0,
        "last_published_snapshot_version": 1,
        "current_learner_update": 1,
        "effective_learner_update": 1,
        "actor_heuristic_fraction_active": 0.0,
        "mirror_mix_fraction_active": 0.0,
        "heuristic_public_mix_fraction_active": 0.0,
        "heuristic_public_variant_mix_fraction_active": 0.0,
        "warmup_snapshot_mix_fraction_active": 0.0,
        "pfsp_pool_size": 0,
        "pfsp_quarantined_opponents": 0,
        "pfsp_champion_pool_size": 0,
        "pfsp_recent_pool_size": 0,
        "pfsp_hard_negative_pool_size": 0,
        "pfsp_last_sampled_envs": 0,
        "pfsp_last_mirror_envs": 0,
        "pfsp_last_heuristic_public_envs": 0,
        "pfsp_last_heuristic_public_variant_envs": 0,
        "pfsp_last_noleague_baseline_envs": 0,
        "pfsp_last_champion_envs": 0,
        "pfsp_last_recent_envs": 0,
        "pfsp_last_hard_negative_envs": 0,
        "pfsp_last_warmup_snapshot_envs": 0,
        "pfsp_epoch": 0,
    }
    kwargs.update(overrides)
    return build_runtime_metrics(**kwargs)
