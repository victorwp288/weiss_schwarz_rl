from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from weiss_rl.replay import runner as replay_runner
from weiss_rl.replay.bundles import ReplayStep, compute_legal_fingerprint64

from .replay_bundle_test_support import FakeReplayEnv, ids_batch, rerun_contract, return_fake_env, write_test_bundle


def test_verify_replay_bundle_replays_actions_and_writes_success_report(tmp_path: Path) -> None:
    contract = rerun_contract()
    spec_hash256 = bytes.fromhex("ab" * 32)
    step0_legal_ids = np.array([1, 4, 9], dtype=np.uint16)
    step1_legal_ids = np.array([2, 5], dtype=np.uint16)
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
        ids_batch(
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
                ids_batch(
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
                ids_batch(
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
        env_factory=lambda observed_contract: return_fake_env(observed_contract, contract, env),
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
