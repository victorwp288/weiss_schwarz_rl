from __future__ import annotations

from weiss_rl.replay.bundles import make_replay_record


def test_make_replay_record_supports_uint64_simulator_episode_keys() -> None:
    record = make_replay_record(
        simulator_episode_key=123456789,
        run_id256=b"r" * 32,
        spec_hash256=b"s" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=4,
        decision_id=5,
        action=6,
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert len(record.episode_key) == 64
    assert record.episode_key64 >= 0
    assert len(record.replay_key256) == 64
    assert record.replay_key64 >= 0
