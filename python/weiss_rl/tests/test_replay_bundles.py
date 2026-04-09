from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.replay import runner as replay_runner
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    load_replay_bundle,
    make_replay_bundle_meta,
    make_replay_record,
    rerun_replay_bundle_fast,
    write_replay_bundle,
)
from weiss_rl.repro import legal_fingerprint_v1


class FakeReplayEnv:
    def __init__(
        self, initial_batch: DecisionBoundaryBatch, transitions: list[tuple[int, DecisionBoundaryBatch]]
    ) -> None:
        self._initial_batch = initial_batch
        self._transitions = list(transitions)
        self.closed = False
        self.reset_seed: int | None = None
        self.actions: list[int] = []

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        self.reset_seed = seed
        return self._initial_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        action = int(np.asarray(actions, dtype=np.int64)[0])
        self.actions.append(action)
        expected_action, next_batch = self._transitions.pop(0)
        assert action == expected_action
        return next_batch

    def close(self) -> None:
        self.closed = True


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
        version=1,
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


def test_rerun_replay_bundle_fast_fails_fast_without_full_rerun_contract(tmp_path: Path) -> None:
    meta = make_replay_bundle_meta(
        simulator_episode_key=99,
        run_id256=b"r" * 32,
        spec_hash256=b"s" * 32,
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
    )
    bundle_path = write_replay_bundle(
        out_dir=tmp_path,
        meta=meta,
        steps=[
            ReplayStep(
                t=0,
                decision_id=1,
                actor=0,
                action=2,
                reward=0.0,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=7,
            )
        ],
    )
    report_path = tmp_path / "replay_verification.json"

    loaded_meta, steps, fault = load_replay_bundle(bundle_path)
    assert loaded_meta.rerun_supported is False
    assert len(steps) == 1
    assert fault is None

    with pytest.raises(RuntimeError, match="full deterministic rerun contract"):
        rerun_replay_bundle_fast(bundle_path=bundle_path, report_path=report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "unsupported"
    assert report["matched"] is False
    assert report["compared_steps"] == 0


def test_rerun_replay_bundle_fast_delegates_to_runner_when_contract_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _rerun_contract()
    bundle_path = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=1,
                actor=0,
                action=2,
                reward=0.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=7,
            )
        ],
    )
    calls: dict[str, Any] = {}

    def fake_verify_replay_bundle(*, bundle_path: Path, report_path: Path | None = None) -> None:
        calls["bundle_path"] = bundle_path
        calls["report_path"] = report_path

    monkeypatch.setattr(replay_runner, "verify_replay_bundle", fake_verify_replay_bundle)
    report_path = tmp_path / "delegated_verification.json"

    rerun_replay_bundle_fast(bundle_path=bundle_path, report_path=report_path)

    assert calls == {"bundle_path": bundle_path, "report_path": report_path}


def test_verify_replay_bundle_replays_actions_and_writes_success_report(tmp_path: Path) -> None:
    contract = _rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    step0_legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    step1_legal_ids = np.array([2, 5], dtype=np.uint16)
    bundle_path = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=4,
                reward=0.25,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=compute_legal_fingerprint64(
                    spec_hash256=spec_hash256,
                    decision_id=10,
                    legal_ids=step0_legal_ids,
                ),
            ),
            ReplayStep(
                t=1,
                decision_id=11,
                actor=1,
                action=5,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=17,
                legal_fingerprint64=compute_legal_fingerprint64(
                    spec_hash256=spec_hash256,
                    decision_id=11,
                    legal_ids=step1_legal_ids,
                ),
            ),
        ],
    )
    report_path = tmp_path / "replay_verification.json"
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=step0_legal_ids,
            episode_seed=44,
            episode_key=555,
        ),
        transitions=[
            (
                4,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=0.25,
                    terminated=False,
                    truncated=False,
                    engine_status=0,
                    legal_ids=step1_legal_ids,
                    episode_seed=44,
                    episode_key=555,
                ),
            ),
            (
                5,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    engine_status=17,
                    legal_ids=np.array([], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                ),
            ),
        ],
    )

    report = replay_runner.verify_replay_bundle(
        bundle_path=bundle_path,
        report_path=report_path,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
    )

    assert report["status"] == "success"
    assert report["matched"] is True
    assert report["compared_steps"] == 2
    assert env.reset_seed == 44
    assert env.actions == [4, 5]
    assert env.closed is True

    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report["status"] == "success"
    assert persisted_report["verified_simulator_episode_key_u64"] == 555


def test_verify_replay_bundle_hard_fails_on_mismatch_and_writes_report(tmp_path: Path) -> None:
    contract = _rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    expected_legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    bundle_path = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=4,
                reward=0.25,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=compute_legal_fingerprint64(
                    spec_hash256=spec_hash256,
                    decision_id=10,
                    legal_ids=expected_legal_ids,
                ),
            )
        ],
    )
    report_path = tmp_path / "replay_verification.json"
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=np.array([1, 4, 8], dtype=np.uint16),
            episode_seed=44,
            episode_key=555,
        ),
        transitions=[],
    )

    with pytest.raises(RuntimeError, match="legal fingerprint mismatch"):
        replay_runner.verify_replay_bundle(
            bundle_path=bundle_path,
            report_path=report_path,
            env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
        )

    assert env.actions == []
    assert env.closed is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "mismatch"
    assert report["matched"] is False
    assert report["compared_steps"] == 0
    assert report["mismatch"]["field"] == "legal_fingerprint64"


def _rerun_contract() -> ReplayRerunContract:
    return ReplayRerunContract(
        version=1,
        observation_visibility="public",
        max_decisions=200,
        max_ticks=10_000,
    )


def _write_bundle(
    tmp_path: Path,
    *,
    contract: ReplayRerunContract | None,
    steps: list[ReplayStep],
) -> Path:
    meta = make_replay_bundle_meta(
        simulator_episode_key=555,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("ab" * 32),
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
        rerun_contract=contract,
    )
    return write_replay_bundle(out_dir=tmp_path, meta=meta, steps=steps)


def _return_fake_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: FakeReplayEnv,
) -> FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


def _ids_batch(
    *,
    decision_id: int,
    actor: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    engine_status: int,
    legal_ids: np.ndarray,
    episode_seed: int,
    episode_key: int,
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.zeros((1, 4), dtype=np.int16),
        reward=np.array([reward], dtype=np.float32),
        terminated=np.array([terminated], dtype=np.bool_),
        truncated=np.array([truncated], dtype=np.bool_),
        to_play=np.array([actor], dtype=np.int32),
        actor=np.array([actor], dtype=np.int32),
        decision_id=np.array([decision_id], dtype=np.int64),
        engine_status=np.array([engine_status], dtype=np.uint8),
        episode_seed=np.array([episode_seed], dtype=np.uint64),
        episode_key=np.array([episode_key], dtype=np.uint64),
        ids_offsets=(ids, np.array([0, int(ids.size)], dtype=np.int32)),
    )
