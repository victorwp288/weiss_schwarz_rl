from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from weiss_rl.replay import runner as replay_runner

from .replay_bundle_test_support import FakeReplayEnv, ids_batch, rerun_contract, return_fake_env, write_test_bundle


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
    contract = rerun_contract()
    bundle_path = write_test_bundle(tmp_path, contract=contract, steps=[])
    report_path = tmp_path / "replay_verification.json"
    env = FakeReplayEnv(
        ids_batch(
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
            env_factory=lambda observed_contract: return_fake_env(observed_contract, contract, env),
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
