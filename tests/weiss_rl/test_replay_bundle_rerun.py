from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from weiss_rl.replay import runner as replay_runner
from weiss_rl.replay.bundles import ReplayStep, load_replay_bundle, rerun_replay_bundle_fast

from .replay_bundle_test_support import rerun_contract, write_test_bundle


def test_rerun_replay_bundle_fast_fails_fast_without_full_rerun_contract(tmp_path: Path) -> None:
    bundle_path = write_test_bundle(
        tmp_path,
        contract=None,
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
    contract = rerun_contract()
    bundle_path = write_test_bundle(
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
