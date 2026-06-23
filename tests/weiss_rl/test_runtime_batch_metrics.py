from __future__ import annotations

import pytest

from .runtime_metrics_test_support import _build_runtime_metrics_with_defaults, _runtime_unroll


def test_build_runtime_metrics_preserves_public_metric_names_and_counter_overrides() -> None:
    metrics, next_cumulative = _build_runtime_metrics_with_defaults(
        selected=[
            _runtime_unroll(
                t=2,
                n=3,
                behavior_policy_version=4,
                counters={
                    "pfsp_sampled_envs": 8,
                    "focal_row_count": 2,
                    "opponent_row_count": 3,
                    "packed_candidate_count": 10,
                    "pfsp_champion_policy_envs__seed_champ_policy_000001": 2,
                    "outcome_v1|w|b1_noleague_baseline": 2,
                    "outcome_v1|l|b1_noleague_baseline": 1,
                    "simulator_step_ns": 2_000_000,
                    "simulator_python_step": 3_000_000,
                },
            ),
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=3,
                counters={"packed_candidate_count": 5, "pass_actions": 2},
            ),
        ],
        occupancy_samples=[0.25, 0.75],
        now=20.0,
        runtime_start=10.0,
        runtime_last_metrics_time=18.0,
        runtime_cumulative_env_steps=10,
        last_published_snapshot_version=5,
        current_learner_update=7,
        effective_learner_update=3,
        actor_heuristic_fraction_active=0.1,
        mirror_mix_fraction_active=0.15,
        heuristic_public_mix_fraction_active=0.2,
        heuristic_public_variant_mix_fraction_active=0.3,
        warmup_snapshot_mix_fraction_active=0.4,
        pfsp_pool_size=11,
        pfsp_quarantined_opponents=12,
        pfsp_champion_pool_size=13,
        pfsp_recent_pool_size=14,
        pfsp_hard_negative_pool_size=15,
        pfsp_last_sampled_envs=16,
        pfsp_last_mirror_envs=17,
        pfsp_last_heuristic_public_envs=18,
        pfsp_last_heuristic_public_variant_envs=19,
        pfsp_last_noleague_baseline_envs=20,
        pfsp_last_champion_envs=21,
        pfsp_last_recent_envs=22,
        pfsp_last_hard_negative_envs=23,
        pfsp_last_warmup_snapshot_envs=24,
        pfsp_epoch=25,
    )

    assert next_cumulative == 17
    assert metrics["batch_env_steps"] == pytest.approx(7.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(3.5)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(1.7)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.5)
    assert metrics["queue_occupancy_p90"] == pytest.approx(0.7)
    assert metrics["policy_version_lag_p50"] == pytest.approx(1.5)
    assert metrics["policy_version_lag_p90"] == pytest.approx(1.9)
    assert metrics["learner_actor_update_lag_p50"] == pytest.approx(3.5)
    assert metrics["learner_actor_update_lag_p90"] == pytest.approx(3.9)
    assert metrics["learner_update_for_collected_batch"] == pytest.approx(7.0)
    assert metrics["last_published_snapshot_version"] == pytest.approx(5.0)
    assert metrics["league_effective_update"] == pytest.approx(3.0)
    assert metrics["league_update_lag"] == pytest.approx(4.0)
    assert metrics["actor_heuristic_fraction_active"] == pytest.approx(0.1)
    assert metrics["mirror_mix_fraction_active"] == pytest.approx(0.15)
    assert metrics["heuristic_public_mix_fraction_active"] == pytest.approx(0.2)
    assert metrics["heuristic_public_variant_mix_fraction_active"] == pytest.approx(0.3)
    assert metrics["warmup_snapshot_mix_fraction_active"] == pytest.approx(0.4)
    assert metrics["pfsp_pool_size"] == pytest.approx(11.0)
    assert metrics["pfsp_quarantined_opponents"] == pytest.approx(12.0)
    assert metrics["pfsp_sampled_envs"] == pytest.approx(8.0)
    assert metrics["pfsp_mirror_envs"] == pytest.approx(17.0)
    assert metrics["pfsp_noleague_baseline_envs"] == pytest.approx(20.0)
    assert metrics["pfsp_warmup_snapshot_envs"] == pytest.approx(24.0)
    assert metrics["pfsp_epoch"] == pytest.approx(25.0)
    assert metrics["collector_pfsp_champion_policy_envs__seed_champ_policy_000001"] == pytest.approx(2.0)
    assert metrics["packed_candidate_count"] == pytest.approx(15.0)
    assert metrics["avg_legal_actions_per_row"] == pytest.approx(3.0)
    assert metrics["collector_pass_actions"] == pytest.approx(2.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_games"] == pytest.approx(3.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_win_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["timer_simulator_step_ms"] == pytest.approx(2.0)
    assert metrics["timer_simulator_python_step_ms"] == pytest.approx(3.0)


def test_build_runtime_metrics_handles_empty_selected_and_empty_occupancy() -> None:
    metrics, next_cumulative = _build_runtime_metrics_with_defaults(
        selected=[],
        occupancy_samples=[],
        now=1.0,
        runtime_start=1.0,
        runtime_last_metrics_time=1.0,
        runtime_cumulative_env_steps=4,
        last_published_snapshot_version=5,
        current_learner_update=2,
        effective_learner_update=5,
        pfsp_pool_size=1,
        pfsp_quarantined_opponents=2,
        pfsp_champion_pool_size=3,
        pfsp_recent_pool_size=4,
        pfsp_hard_negative_pool_size=5,
        pfsp_last_sampled_envs=6,
        pfsp_last_mirror_envs=7,
        pfsp_last_heuristic_public_envs=8,
        pfsp_last_heuristic_public_variant_envs=9,
        pfsp_last_noleague_baseline_envs=10,
        pfsp_last_champion_envs=11,
        pfsp_last_recent_envs=12,
        pfsp_last_hard_negative_envs=13,
        pfsp_last_warmup_snapshot_envs=14,
        pfsp_epoch=15,
    )

    assert next_cumulative == 4
    assert metrics["batch_env_steps"] == pytest.approx(0.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(0.0)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(4_000_000.0)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.0)
    assert metrics["policy_version_lag_p50"] == pytest.approx(0.0)
    assert metrics["learner_actor_update_lag_p50"] == pytest.approx(0.0)
    assert metrics["league_update_lag"] == pytest.approx(0.0)
    assert metrics["pfsp_sampled_envs"] == pytest.approx(6.0)
    assert metrics["pfsp_hard_negative_envs"] == pytest.approx(13.0)
    assert metrics["avg_legal_actions_per_row"] == pytest.approx(0.0)
