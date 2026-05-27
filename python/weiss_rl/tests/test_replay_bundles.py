from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weiss_rl.artifacts.reproducibility import legal_fingerprint_v1
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


@pytest.mark.parametrize(
    ("batch_kwargs", "error_match", "mismatch_field", "expected_value", "observed_value"),
    [
        (
            {"episode_seed": 45, "episode_key": 555},
            "Replay reset seed mismatch",
            "episode_seed64",
            44,
            45,
        ),
        (
            {"episode_seed": 44, "episode_key": 556},
            "Replay reset episode_key mismatch",
            "simulator_episode_key_u64",
            555,
            556,
        ),
    ],
)
def test_verify_replay_bundle_reports_reset_identity_mismatches_as_structured_mismatches(
    tmp_path: Path,
    batch_kwargs: dict[str, int],
    error_match: str,
    mismatch_field: str,
    expected_value: int,
    observed_value: int,
) -> None:
    contract = _rerun_contract()
    bundle_path = _write_bundle(tmp_path, contract=contract, steps=[])
    report_path = tmp_path / "replay_verification.json"
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=np.array([1, 4, 9], dtype=np.uint16),
            episode_seed=batch_kwargs["episode_seed"],
            episode_key=batch_kwargs["episode_key"],
        ),
        transitions=[],
    )

    with pytest.raises(RuntimeError, match=error_match):
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
    assert report["mismatch"] == {
        "field": mismatch_field,
        "expected": expected_value,
        "observed": observed_value,
    }


def test_verify_replay_bundle_rejects_unsupported_rerun_contract_versions_before_running_env(
    tmp_path: Path,
) -> None:
    contract = _rerun_contract(version=99)
    bundle_path = _write_bundle(tmp_path, contract=contract, steps=[])
    report_path = tmp_path / "replay_verification.json"
    env_factory_called = False

    def unexpected_env_factory(_: ReplayRerunContract) -> FakeReplayEnv:
        nonlocal env_factory_called
        env_factory_called = True
        raise AssertionError("env_factory should not be called for unsupported rerun contract versions")

    with pytest.raises(RuntimeError, match="unsupported: expected version 2, got 99"):
        replay_runner.verify_replay_bundle(
            bundle_path=bundle_path,
            report_path=report_path,
            env_factory=unexpected_env_factory,
        )

    assert env_factory_called is False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "unsupported"
    assert report["matched"] is False
    assert report["compared_steps"] == 0
    assert report["unsupported_rerun_contract_version"] == 99


def test_verify_replay_bundle_verifies_recorded_step_index(tmp_path: Path) -> None:
    contract = _rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    bundle_path = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=7,
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
                    legal_ids=legal_ids,
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
            legal_ids=legal_ids,
            episode_seed=44,
            episode_key=555,
        ),
        transitions=[],
    )

    with pytest.raises(RuntimeError, match="Replay step index mismatch"):
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
    assert report["mismatch"] == {
        "field": "t",
        "expected": 7,
        "observed": 0,
    }


def test_build_replay_env_uses_fast_pool_factory_for_default_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _rerun_contract()
    pool = object()
    pool_factory_calls: dict[str, Any] = {}
    wrapper_calls: dict[str, Any] = {}

    class FakeDecisionBoundaryEnv:
        def __init__(self, pool_arg: Any, **kwargs: Any) -> None:
            wrapper_calls["pool"] = pool_arg
            wrapper_calls.update(kwargs)

    def fake_make_env_pool_from_config(
        env_config: dict[str, Any],
        *,
        profile: str,
        num_envs: int | None = None,
    ) -> tuple[object, str]:
        pool_factory_calls["env_config"] = dict(env_config)
        pool_factory_calls["profile"] = profile
        pool_factory_calls["num_envs"] = num_envs
        return pool, "i16_legal_ids"

    monkeypatch.setattr(replay_runner, "make_env_pool_from_config", fake_make_env_pool_from_config)
    monkeypatch.setattr(replay_runner, "DecisionBoundaryEnv", FakeDecisionBoundaryEnv)

    env = replay_runner.build_replay_env(contract)

    assert isinstance(env, FakeDecisionBoundaryEnv)
    assert pool_factory_calls == {
        "env_config": {
            "max_decisions": contract.max_decisions,
            "max_ticks": contract.max_ticks,
            "observation_visibility": contract.observation_visibility,
            "seed": 0,
            "reward_json": contract.reward_json,
            "curriculum_json": contract.curriculum_json,
            "deck": contract.deck,
            "opponent_deck": contract.opponent_deck,
        },
        "profile": "fast",
        "num_envs": 1,
    }
    assert wrapper_calls["pool"] is pool
    assert wrapper_calls["legality"] == "ids_offsets"
    assert wrapper_calls["engine_status_policy"] == "passthrough"
    assert pool_factory_calls["env_config"]["reward_json"] == contract.reward_json
    assert pool_factory_calls["env_config"]["curriculum_json"] == contract.curriculum_json


def _rerun_contract(
    *,
    version: int = 2,
    reward_json: str | None = '{"objective":"terminal_pm1"}',
    curriculum_json: str | None = '{"version":"curriculum_v1"}',
    deck: str | None = "preset:main_deck_5hy_yotsuba_v1",
    opponent_deck: str | None = "preset:main_deck_5hy_yotsuba_v1",
) -> ReplayRerunContract:
    return ReplayRerunContract(
        version=version,
        observation_visibility="public",
        max_decisions=200,
        max_ticks=10_000,
        reward_json=reward_json,
        curriculum_json=curriculum_json,
        deck=deck,
        opponent_deck=opponent_deck,
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
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([episode_seed], dtype=np.uint64),
        episode_key=np.array([episode_key], dtype=np.uint64),
        ids_offsets=(ids, np.array([0, int(ids.size)], dtype=np.int32)),
    )
