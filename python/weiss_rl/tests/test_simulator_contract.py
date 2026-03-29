from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.repro import canonical_json_bytes, sha256_hex
from weiss_rl.simulator_contract import _ProbeTarget, load_simulator_contract


def _nested_spec_bundle(*, spec_hash: int = 123) -> dict[str, object]:
    return {
        "policy_version": 3,
        "spec_hash": spec_hash,
        "observation": {
            "obs_encoding_version": 2,
            "dtype": "i32",
            "obs_len": 4,
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 7,
            "pass_action_id": 1,
        },
    }


def test_load_simulator_contract_uses_installed_package_first(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "simulator": {"version": "0.7.0", "build_info": {"profile": "release"}},
        "spec_bundle": _nested_spec_bundle(),
    }

    monkeypatch.setattr(
        "weiss_rl.simulator_contract._candidate_targets",
        lambda repo_root: [_ProbeTarget(python="/good/python")],
    )
    monkeypatch.setattr("weiss_rl.simulator_contract._run_probe", lambda target: payload)

    contract = load_simulator_contract(Path("/repo"))

    assert contract.simulator["version"] == "0.7.0"
    assert contract.simulator["compatibility_hash"] == "123"
    assert contract.spec_bundle == payload["spec_bundle"]
    assert contract.spec_hash256 == sha256_hex(canonical_json_bytes(payload["spec_bundle"]))


def test_load_simulator_contract_uses_first_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    targets = [
        _ProbeTarget(python="/bad/python"),
        _ProbeTarget(python="/good/python", pythonpath=tmp_path / "sim"),
    ]
    payload = {
        "simulator": {"version": "0.7.0", "build_info": {"profile": "release"}},
        "spec_bundle": _nested_spec_bundle(),
    }

    monkeypatch.setattr("weiss_rl.simulator_contract._candidate_targets", lambda repo_root: targets)

    def fake_run_probe(target: _ProbeTarget) -> dict[str, object]:
        if target.python == "/bad/python":
            raise OSError("boom")
        return payload

    monkeypatch.setattr("weiss_rl.simulator_contract._run_probe", fake_run_probe)

    contract = load_simulator_contract(Path("/repo"))

    assert contract.simulator["version"] == "0.7.0"
    assert contract.simulator["compatibility_hash"] == "123"
    assert contract.spec_bundle == payload["spec_bundle"]
    assert contract.spec_hash256 == sha256_hex(canonical_json_bytes(payload["spec_bundle"]))


def test_load_simulator_contract_skips_invalid_spec_bundle_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    targets = [
        _ProbeTarget(python="/bad/python"),
        _ProbeTarget(python="/good/python", pythonpath=tmp_path / "sim"),
    ]
    payload = {
        "simulator": {"version": "0.7.0", "build_info": {"profile": "release"}},
        "spec_bundle": {
            "spec_hash": 123,
            "observation": {"obs_encoding_version": 2, "dtype": "i32", "obs_len": 4},
            "action": {"action_encoding_version": 1, "pass_action_id": 1},
        },
    }

    monkeypatch.setattr("weiss_rl.simulator_contract._candidate_targets", lambda repo_root: targets)
    monkeypatch.setattr("weiss_rl.simulator_contract._run_probe", lambda target: payload)

    with pytest.raises(RuntimeError, match="invalid spec_bundle payload"):
        load_simulator_contract(Path("/repo"))


def test_load_simulator_contract_raises_after_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.simulator_contract._candidate_targets", lambda repo_root: [])

    with pytest.raises(RuntimeError, match="WEISS_SIM_PYTHONPATH"):
        load_simulator_contract(Path("/repo"))
