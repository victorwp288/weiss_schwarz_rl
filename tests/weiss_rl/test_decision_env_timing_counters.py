from __future__ import annotations

from weiss_rl.envs.decision_env import DecisionBoundaryEnv

from tests.weiss_rl.decision_env_test_support import FakePool


def test_decision_boundary_env_drains_simulator_timing_counters() -> None:
    pool = FakePool()
    pool.timing_snapshot.update(
        {
            "timing_enabled": True,
            "step_sample_from_logits_with_logp_into_i16_legal_ids_count": 2,
            "step_sample_from_logits_with_logp_into_i16_legal_ids_ns": 1500,
            "legal_action_meta_materialize_count": 3,
            "legal_action_meta_materialize_ns": 2750,
        }
    )

    env = DecisionBoundaryEnv(pool, legality="ids_offsets", profile_timers=True)
    counters = env.drain_timing_counters()

    assert pool.timing_enabled is True
    assert pool.timing_reset_calls == 2
    assert counters == {
        "step_sample_from_logits_with_logp_into_i16_legal_ids_count": 2,
        "step_sample_from_logits_with_logp_into_i16_legal_ids_ns": 1500,
        "legal_action_meta_materialize_count": 3,
        "legal_action_meta_materialize_ns": 2750,
    }
