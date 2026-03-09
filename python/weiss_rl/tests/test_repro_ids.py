from __future__ import annotations

from weiss_rl.repro import (
    derive_actor_seed,
    derive_episode_key256,
    derive_episode_seed,
    key256_to_short64,
    normalize_simulator_episode_key256,
    resolve_episode_key256,
)


def test_seed_derivation_is_deterministic() -> None:
    actor_seed_a = derive_actor_seed(20260212, actor_id=4)
    actor_seed_b = derive_actor_seed(20260212, actor_id=4)
    assert actor_seed_a == actor_seed_b

    episode_seed_a = derive_episode_seed(actor_seed_a, env_id=1, episode_index=77)
    episode_seed_b = derive_episode_seed(actor_seed_b, env_id=1, episode_index=77)
    assert episode_seed_a == episode_seed_b


def test_normalize_simulator_episode_key256_accepts_uint64_keys() -> None:
    key_a = normalize_simulator_episode_key256(123456789)
    key_b = normalize_simulator_episode_key256(123456789)

    assert key_a == key_b
    assert len(key_a) == 32


def test_normalize_simulator_episode_key256_accepts_non_32_byte_raw_keys() -> None:
    key = normalize_simulator_episode_key256(b"raw-sim-key")

    assert len(key) == 32


def test_resolve_episode_key256_prefers_simulator_key_over_fallback() -> None:
    simulator_key = (b"sim-episode-key-32-bytes-long!!!")[:32]

    resolved = resolve_episode_key256(
        simulator_episode_key=simulator_key,
        run_id256=b"r" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=4,
    )

    assert resolved == simulator_key
    assert key256_to_short64(resolved) == int.from_bytes(simulator_key[:8], byteorder="little", signed=False)


def test_resolve_episode_key256_derives_fallback_when_simulator_key_missing() -> None:
    resolved = resolve_episode_key256(
        simulator_episode_key=None,
        run_id256=b"r" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=4,
    )

    assert len(resolved) == 32


def test_resolve_episode_key256_treats_empty_bytes_as_missing() -> None:
    resolved = resolve_episode_key256(
        simulator_episode_key=b"",
        run_id256=b"r" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=4,
    )

    expected = derive_episode_key256(
        run_id256=b"r" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=4,
    )

    assert resolved == expected
