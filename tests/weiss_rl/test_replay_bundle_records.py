from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.artifacts.reproducibility import legal_fingerprint_v1
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    make_replay_record,
)


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


def test_compute_legal_fingerprint64_uses_canonical_contract() -> None:
    spec_hash256 = bytes.fromhex("ab" * 32)
    legal_ids = np.array([1, 3, 9], dtype=np.uint16)

    assert compute_legal_fingerprint64(
        spec_hash256=spec_hash256,
        decision_id=7,
        legal_ids=legal_ids,
    ) == legal_fingerprint_v1(spec_hash256, decision_id=7, legal_ids=legal_ids)


def test_compute_legal_fingerprint64_rejects_unsorted_legal_ids() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_legal_fingerprint64(
            spec_hash256=bytes.fromhex("cd" * 32),
            decision_id=11,
            legal_ids=np.array([1, 3, 3], dtype=np.uint16),
        )


def test_make_replay_bundle_meta_preserves_simulator_episode_identity() -> None:
    meta = make_replay_bundle_meta(
        simulator_episode_key=1234,
        run_id256=b"r" * 32,
        spec_hash256=b"s" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
    )

    assert meta.schema_version == 2
    assert meta.episode_identity_source == "simulator"
    assert meta.simulator_episode_key_kind == "u64"
    assert meta.simulator_episode_key_u64 == 1234
    assert meta.simulator_episode_key_hex is None
    assert meta.episode_seed64 == 44
    assert meta.rerun_contract is None
    assert meta.rerun_supported is False
    assert meta.rerun_blocker is not None


def test_make_replay_bundle_meta_marks_bundle_rerunnable_when_contract_present() -> None:
    contract = ReplayRerunContract(
        version=2,
        observation_visibility="public",
        max_decisions=200,
        max_ticks=10_000,
    )
    meta = make_replay_bundle_meta(
        simulator_episode_key=1234,
        run_id256=b"r" * 32,
        spec_hash256=b"s" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
        rerun_contract=contract,
    )

    assert meta.schema_version == 3
    assert meta.rerun_contract == contract
    assert meta.rerun_supported is True
    assert meta.rerun_blocker is None
