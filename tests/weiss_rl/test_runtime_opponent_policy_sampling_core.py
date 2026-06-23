from __future__ import annotations

from .runtime_opponent_sampling_call_test_support import sample_runtime_policy_ids, sampling_config


def test_sample_runtime_opponent_policy_ids_handles_empty_and_no_league_cases() -> None:
    empty = sample_runtime_policy_ids(count=0, rng_seed=1, league_enabled=False)
    no_league = sample_runtime_policy_ids(count=3, rng_seed=1, league_enabled=False)

    assert empty.policy_ids == ()
    assert empty.sampled_envs == 0
    assert no_league.policy_ids == ("mirror", "mirror", "mirror")
    assert no_league.sampled_envs == 0
    assert no_league.mirror_envs == 3


def test_sample_runtime_opponent_policy_ids_preserves_pre_pfsp_mixed_bucket_accounting() -> None:
    result = sample_runtime_policy_ids(
        count=12,
        rng_seed=1,
        league_config=sampling_config(warmup_first_updates=999),
        pfsp_ready=False,
        heuristic_public_weight=0.2,
        heuristic_public_variant_weight=0.2,
        noleague_baseline_weight=0.2,
        warmup_snapshot_weight=0.2,
        opponent_candidate_ids=("warm_a", "warm_b"),
        opponent_heuristic_policy_ids=("heuristic", "aggro", "control"),
        opponent_model_ids=("baseline", "warm_a", "warm_b"),
    )

    assert result.policy_ids == (
        "baseline",
        "mirror",
        "heuristic",
        "mirror",
        "control",
        "baseline",
        "mirror",
        "baseline",
        "baseline",
        "heuristic",
        "warm_a",
        "baseline",
    )
    assert result.sampled_envs == 9
    assert result.mirror_envs == 3
    assert result.heuristic_public_envs == 2
    assert result.heuristic_public_variant_envs == 1
    assert result.noleague_baseline_envs == 5
    assert result.warmup_snapshot_envs == 1


def test_sample_runtime_opponent_policy_ids_supports_live_mirror_lane_after_pfsp_ready() -> None:
    result = sample_runtime_policy_ids(
        count=200,
        rng_seed=11,
        league_config=sampling_config(),
        pfsp_ready=True,
        reference_update=10,
        mirror_weight=0.4,
        heuristic_public_weight=0.2,
        opponent_candidate_ids=("recent_a", "recent_b"),
        opponent_recent_ids=("recent_a", "recent_b"),
        opponent_heuristic_policy_ids=("heuristic",),
        opponent_model_ids=("recent_a", "recent_b"),
    )

    assert len(result.policy_ids) == 200
    assert result.mirror_envs > 0
    assert result.heuristic_public_envs > 0
    assert result.recent_envs > 0
    assert result.warmup_snapshot_envs == 0
    assert result.sampled_envs == result.heuristic_public_envs + result.recent_envs
