from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.replay import runner as replay_runner
from weiss_rl.replay.bundles import ReplayRerunContract

from .replay_bundle_test_support import FakeReplayEnv, rerun_contract, write_test_bundle


def test_verify_replay_bundle_rejects_unsupported_rerun_contract_versions_before_running_env(
    tmp_path: Path,
) -> None:
    contract = rerun_contract(version=99)
    bundle_path = write_test_bundle(tmp_path, contract=contract, steps=[])
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
