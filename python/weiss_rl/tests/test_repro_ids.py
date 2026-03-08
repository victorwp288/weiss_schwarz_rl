from __future__ import annotations

from weiss_rl.repro import (
    compute_run_id256,
    compute_run_id64,
    derive_actor_seed,
    derive_episode_seed,
    serialize_run_identity,
    stable_hash64,
)


def test_seed_derivation_is_deterministic() -> None:
    actor_seed_a = derive_actor_seed(20260212, actor_id=4)
    actor_seed_b = derive_actor_seed(20260212, actor_id=4)
    assert actor_seed_a == actor_seed_b

    episode_seed_a = derive_episode_seed(actor_seed_a, env_id=1, episode_index=77)
    episode_seed_b = derive_episode_seed(actor_seed_b, env_id=1, episode_index=77)
    assert episode_seed_a == episode_seed_b



def test_run_id_serialization_is_tagged_and_stable() -> None:
    spec_hash = "00" * 32
    config_hash = "11" * 32
    git_commit = "0123456789abcdef0123456789abcdef01234567"
    start_nonce = 42

    payload = serialize_run_identity(spec_hash, config_hash, git_commit, start_nonce)
    expected = b"".join(
        (
            (3).to_bytes(4, "little") + b"run" + (0).to_bytes(4, "little"),
            (4).to_bytes(4, "little") + b"spec" + (32).to_bytes(4, "little") + bytes.fromhex(spec_hash),
            (6).to_bytes(4, "little") + b"config" + (32).to_bytes(4, "little") + bytes.fromhex(config_hash),
            (3).to_bytes(4, "little") + b"git" + (20).to_bytes(4, "little") + bytes.fromhex(git_commit),
            (5).to_bytes(4, "little") + b"nonce" + (8).to_bytes(4, "little") + start_nonce.to_bytes(8, "little"),
        )
    )

    assert payload == expected
    assert compute_run_id256(spec_hash, config_hash, git_commit, start_nonce) == (
        "ebfe79ae78c3b0080ecf5c46e5118d6dbc1435a96067001b27039cc7d1abbdcf"
    )
    assert compute_run_id64(spec_hash, config_hash, git_commit, start_nonce) == stable_hash64(expected)
