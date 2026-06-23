from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from weiss_rl.replay import runner as replay_runner
from weiss_rl.replay.bundles import ReplayStep, compute_legal_fingerprint64

from .replay_bundle_test_support import FakeReplayEnv, ids_batch, rerun_contract, return_fake_env, write_test_bundle


def test_verify_replay_bundle_hard_fails_on_mismatch_and_writes_report(tmp_path: Path) -> None:
    contract = rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    expected_legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    bundle_path = write_test_bundle(
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
        ids_batch(
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
            env_factory=lambda observed_contract: return_fake_env(observed_contract, contract, env),
        )

    assert env.actions == []
    assert env.closed is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "mismatch"
    assert report["matched"] is False
    assert report["compared_steps"] == 0
    assert report["mismatch"]["field"] == "legal_fingerprint64"


def test_verify_replay_bundle_verifies_recorded_step_index(tmp_path: Path) -> None:
    contract = rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    bundle_path = write_test_bundle(
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
        ids_batch(
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
            env_factory=lambda observed_contract: return_fake_env(observed_contract, contract, env),
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
