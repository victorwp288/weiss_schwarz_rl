"""Initial opponent and PFSP bookkeeping state for QueueRuntime."""

from __future__ import annotations

from typing import Any

from weiss_rl.league.outcomes import OnlineOutcomeTracker


def initialize_runtime_opponent_state(runtime: Any, *, league_config: Any | None) -> None:
    runtime._opponent_sampler = None
    runtime._opponent_candidate_ids = ()
    runtime._outcomes = OnlineOutcomeTracker(
        window_size=(50_000 if league_config is None else int(league_config.pfsp_window_episodes))
    )
    runtime._pfsp_epoch = int(runtime._outcomes.current_epoch)
    runtime._current_learner_update = 0
    runtime._effective_learner_update = 0
    runtime._published_snapshot_update_by_fingerprint = {}
    runtime._pfsp_pool_size = 0
    runtime._pfsp_quarantined_opponents = 0
    runtime._pfsp_champion_pool_size = 0
    runtime._pfsp_recent_pool_size = 0
    runtime._pfsp_hard_negative_pool_size = 0
    runtime._pfsp_last_sampled_envs = 0
    runtime._pfsp_last_mirror_envs = 0
    runtime._pfsp_last_heuristic_public_envs = 0
    runtime._pfsp_last_heuristic_public_variant_envs = 0
    runtime._pfsp_last_noleague_baseline_envs = 0
    runtime._pfsp_last_champion_envs = 0
    runtime._pfsp_last_recent_envs = 0
    runtime._pfsp_last_hard_negative_envs = 0
    runtime._pfsp_last_warmup_snapshot_envs = 0
    runtime._pfsp_last_sampled_policy_envs = {}
    runtime._pfsp_last_heuristic_public_policy_envs = {}
    runtime._pfsp_last_heuristic_public_variant_policy_envs = {}
    runtime._pfsp_last_noleague_baseline_policy_envs = {}
    runtime._pfsp_last_champion_policy_envs = {}
    runtime._pfsp_last_recent_policy_envs = {}
    runtime._pfsp_last_hard_negative_policy_envs = {}
    runtime._pfsp_last_warmup_snapshot_policy_envs = {}
    runtime._disable_mirror_policy_fusion = False
    runtime._opponent_champion_ids = ()
    runtime._opponent_recent_ids = ()
    runtime._opponent_hard_negative_ids = ()


__all__ = ["initialize_runtime_opponent_state"]
