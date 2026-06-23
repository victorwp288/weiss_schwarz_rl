from __future__ import annotations

from pathlib import Path

import pytest
import torch
from weiss_rl.training.checkpointing.load import (
    load_initialization_checkpoint_contract,
    load_resume_checkpoint_contract,
)
from weiss_rl.training.checkpointing.restore import (
    validate_checkpoint_payload_contract,
    warn_if_config_hash_mismatch_allowed,
)


def test_validate_checkpoint_payload_contract_reports_allowed_config_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "format": "minimal_train_checkpoint_v1",
        "config_hash256": "old",
        "spec_hash256": "spec",
        "algorithm": "impala_vtrace_gru",
        "model_state_dict": {},
    }

    contract = validate_checkpoint_payload_contract(
        payload,
        checkpoint_path=tmp_path / "checkpoint.pt",
        expected_config_hash="new",
        expected_spec_hash256="spec",
        algorithm="impala_vtrace_gru",
        allow_config_mismatch=True,
    )
    warn_if_config_hash_mismatch_allowed(contract)

    assert contract.config_hash_mismatch is True
    assert "allowing checkpoint config hash mismatch" in capsys.readouterr().err


def test_load_resume_checkpoint_contract_uses_unsafe_torch_load_and_warns_on_allowed_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    payload = {
        "format": "minimal_train_checkpoint_v1",
        "config_hash256": "old-config",
        "spec_hash256": "spec",
        "algorithm": "impala_vtrace_gru",
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }
    calls: list[tuple[Path, torch.device, bool]] = []

    def fake_torch_load(path: Path, *, map_location: torch.device, weights_only: bool) -> dict[str, object]:
        calls.append((path, map_location, weights_only))
        return payload

    monkeypatch.setattr("weiss_rl.training.checkpointing.load.torch.load", fake_torch_load)

    contract = load_resume_checkpoint_contract(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        expected_config_hash="new-config",
        expected_spec_hash256="spec",
        algorithm="impala_vtrace_gru",
        allow_config_mismatch=True,
    )

    assert calls == [(checkpoint_path, torch.device("cpu"), False)]
    assert contract.payload is payload
    assert contract.config_hash_mismatch is True
    assert "allowing checkpoint config hash mismatch" in capsys.readouterr().err


def test_load_initialization_checkpoint_contract_allows_config_mismatch_without_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "format": "minimal_train_checkpoint_v1",
        "config_hash256": "source-config",
        "spec_hash256": "spec",
        "algorithm": "impala_vtrace_ff",
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }
    monkeypatch.setattr("weiss_rl.training.checkpointing.load.torch.load", lambda *_args, **_kwargs: payload)

    contract = load_initialization_checkpoint_contract(
        checkpoint_path=tmp_path / "checkpoint.pt",
        device=torch.device("cpu"),
        expected_spec_hash256="spec",
        algorithm="impala_vtrace_ff",
    )

    assert contract.payload is payload
    assert contract.config_hash_mismatch is True
    assert capsys.readouterr().err == ""
