from __future__ import annotations

from types import SimpleNamespace

from weiss_rl.runtime.components.opponent_startup import initialize_runtime_opponent_state


def test_initialize_runtime_opponent_state_sets_default_counters() -> None:
    runtime = SimpleNamespace()

    initialize_runtime_opponent_state(runtime, league_config=None)

    assert runtime._opponent_sampler is None
    assert runtime._opponent_candidate_ids == ()
    assert runtime._outcomes.current_epoch == 0
    assert runtime._current_learner_update == 0
    assert runtime._effective_learner_update == 0
    assert runtime._published_snapshot_update_by_fingerprint == {}
    assert runtime._pfsp_pool_size == 0
    assert runtime._pfsp_quarantined_opponents == 0
    assert runtime._pfsp_champion_pool_size == 0
    assert runtime._pfsp_recent_pool_size == 0
    assert runtime._pfsp_hard_negative_pool_size == 0
    assert runtime._pfsp_last_sampled_envs == 0
    assert runtime._pfsp_last_mirror_envs == 0
    assert runtime._pfsp_last_heuristic_public_envs == 0
    assert runtime._pfsp_last_heuristic_public_variant_envs == 0
    assert runtime._pfsp_last_noleague_baseline_envs == 0
    assert runtime._pfsp_last_champion_envs == 0
    assert runtime._pfsp_last_recent_envs == 0
    assert runtime._pfsp_last_hard_negative_envs == 0
    assert runtime._pfsp_last_warmup_snapshot_envs == 0
    assert runtime._pfsp_last_sampled_policy_envs == {}
    assert runtime._pfsp_last_heuristic_public_policy_envs == {}
    assert runtime._pfsp_last_heuristic_public_variant_policy_envs == {}
    assert runtime._pfsp_last_noleague_baseline_policy_envs == {}
    assert runtime._pfsp_last_champion_policy_envs == {}
    assert runtime._pfsp_last_recent_policy_envs == {}
    assert runtime._pfsp_last_hard_negative_policy_envs == {}
    assert runtime._pfsp_last_warmup_snapshot_policy_envs == {}
    assert runtime._disable_mirror_policy_fusion is False
    assert runtime._opponent_champion_ids == ()
    assert runtime._opponent_recent_ids == ()
    assert runtime._opponent_hard_negative_ids == ()


def test_initialize_runtime_opponent_state_uses_league_pfsp_window() -> None:
    runtime = SimpleNamespace()

    initialize_runtime_opponent_state(runtime, league_config=SimpleNamespace(pfsp_window_episodes=123))

    assert runtime._outcomes.window_size == 123
