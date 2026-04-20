from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.repro import (
    compute_run_id64,
    compute_run_id256,
    derive_actor_seed,
    derive_episode_key256,
    derive_episode_seed,
    hash_seed_file,
    key256_to_short64,
    legal_fingerprint_v1,
    normalize_simulator_episode_key256,
    parse_seed_file,
    resolve_episode_key256,
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


def test_legal_fingerprint_is_deterministic() -> None:
    spec_hash = bytes.fromhex("ab" * 32)
    fingerprint_a = legal_fingerprint_v1(spec_hash, decision_id=7, legal_ids=[1, 3, 9])
    fingerprint_b = legal_fingerprint_v1(spec_hash, decision_id=7, legal_ids=[1, 3, 9])
    assert fingerprint_a == fingerprint_b


def test_legal_fingerprint_rejects_unsorted_legal_ids() -> None:
    spec_hash = bytes.fromhex("cd" * 32)
    with pytest.raises(ValueError, match="strictly increasing"):
        legal_fingerprint_v1(spec_hash, decision_id=7, legal_ids=[1, 3, 3])


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


def test_parse_seed_file_valid(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n456\n789\n")
    seeds = parse_seed_file(seed_file)
    assert seeds == [123, 456, 789]


def test_parse_seed_file_rejects_blank_lines(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n\n456\n")

    with pytest.raises(ValueError, match="Blank line 2"):
        parse_seed_file(seed_file)


def test_parse_seed_file_rejects_comments(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n# comment\n456\n")

    with pytest.raises(ValueError, match="Comment on line 2"):
        parse_seed_file(seed_file)


def test_parse_seed_file_invalid_line(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\nabc\n456\n")

    with pytest.raises(ValueError, match="Invalid seed on line 2"):
        parse_seed_file(seed_file)


def test_parse_seed_file_out_of_range(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n-1\n456\n")

    with pytest.raises(ValueError, match="out of u64 range"):
        parse_seed_file(seed_file)


def test_hash_seed_file_rejects_invalid_contents(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n# comment\n456\n")

    with pytest.raises(ValueError, match="Comment on line 2"):
        hash_seed_file(seed_file)


def test_hash_seed_file_is_stable_across_lf_and_crlf(tmp_path: Path) -> None:
    lf_seed_file = tmp_path / "seeds_lf.txt"
    crlf_seed_file = tmp_path / "seeds_crlf.txt"

    lf_seed_file.write_bytes(b"123\n456\n789\n")
    crlf_seed_file.write_bytes(b"123\r\n456\r\n789\r\n")

    assert hash_seed_file(lf_seed_file) == hash_seed_file(crlf_seed_file)


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
