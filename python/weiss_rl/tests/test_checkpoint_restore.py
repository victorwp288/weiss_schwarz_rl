from __future__ import annotations

from pathlib import Path

import pytest
import torch

from weiss_rl.training.checkpointing.load import (
    load_initialization_checkpoint_contract,
    load_resume_checkpoint_contract,
)
from weiss_rl.training.checkpointing.restore import (
    CheckpointPayloadContract,
    apply_minimal_checkpoint_initialization,
    apply_minimal_checkpoint_resume_state,
    validate_checkpoint_payload_contract,
    warn_if_config_hash_mismatch_allowed,
)
from weiss_rl.training.checkpointing.restore_state import (
    apply_checkpoint_resume_counters,
    checkpoint_counter_state_from_payload,
)


class _Model:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.loaded_state = state_dict


class _Optimizer:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


class _GradScaler:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


class _OrderedModel:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._events.append("model")
        self.loaded_state = state_dict


class _OrderedOptimizer:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self._events.append("optimizer")
        self.loaded_state = state_dict


class _OrderedGradScaler:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self._events.append("grad_scaler")
        self.loaded_state = state_dict


class _ResumeLearner:
    def __init__(self) -> None:
        self.model = _Model()
        self.optimizer = _Optimizer()
        self._grad_scaler = _GradScaler()
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> _Optimizer:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.loaded_anchor_state = state_dict


class _OrderedResumeLearner:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.model = _OrderedModel(self.events)
        self.optimizer = _OrderedOptimizer(self.events)
        self._grad_scaler = _OrderedGradScaler(self.events)
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> _OrderedOptimizer:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.events.append("anchor")
        self.loaded_anchor_state = state_dict


class _InitLearnerWithoutReset:
    def __init__(self) -> None:
        self.model = _Model()
        self.optimizer = _Optimizer()
        self._grad_scaler = _GradScaler()
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> _Optimizer:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.loaded_anchor_state = state_dict


def _contract(payload: dict[str, object] | None = None) -> CheckpointPayloadContract:
    full_payload: dict[str, object] = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": 3,
        "policy_version": 7,
        "total_samples_processed": 42,
        "init_schedule_offset_updates": 5,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "policy_anchor_model_state_dict": {"anchor_weight": torch.tensor([2.0])},
        "optimizer_state_dict": {"lr": 0.01},
        "grad_scaler_state_dict": {"scale": 2.0},
    }
    if payload is not None:
        full_payload.update(payload)
    return CheckpointPayloadContract(
        payload=full_payload,
        model_state_dict=full_payload["model_state_dict"],  # type: ignore[arg-type]
        config_hash_mismatch=False,
        expected_config_hash="abc",
        payload_config_hash="abc",
    )


def test_apply_minimal_checkpoint_resume_state_restores_training_state(tmp_path: Path) -> None:
    learner = _ResumeLearner()
    guidance_calls: list[tuple[object, dict[str, object]]] = []

    resume = apply_minimal_checkpoint_resume_state(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=_contract(),
        restore_model_guidance=lambda model, payload: guidance_calls.append((model, payload)),
        restore_counters=True,
    )

    assert resume.update_count == 3
    assert resume.policy_version == 7
    assert resume.total_samples_processed == 42
    assert resume.init_schedule_offset_updates == 5
    assert learner.update_count == 3
    assert learner.policy_version == 7
    assert learner.total_samples_processed == 42
    assert learner.init_schedule_offset_updates == 5
    assert learner.model.loaded_state is not None
    assert learner.optimizer.loaded_state == {"lr": 0.01}
    assert learner._grad_scaler.loaded_state == {"scale": 2.0}
    assert learner.loaded_anchor_state is not None
    assert guidance_calls[0][0] is learner.model


def test_apply_minimal_checkpoint_resume_state_preserves_restore_order_and_counters(
    tmp_path: Path,
) -> None:
    learner = _OrderedResumeLearner()

    resume = apply_minimal_checkpoint_resume_state(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=_contract(),
        restore_model_guidance=lambda _model, _payload: learner.events.append("guidance"),
        restore_counters=False,
    )

    assert learner.events == ["model", "guidance", "anchor", "optimizer", "grad_scaler"]
    assert resume.update_count == 99
    assert resume.policy_version == 88
    assert resume.total_samples_processed == 77
    assert resume.init_schedule_offset_updates == 5
    assert learner.update_count == 99
    assert learner.policy_version == 88
    assert learner.total_samples_processed == 77
    assert learner.init_schedule_offset_updates == 5
    assert learner.start_time == 0.0


def test_apply_minimal_checkpoint_resume_state_rejects_invalid_anchor_state(tmp_path: Path) -> None:
    learner = _ResumeLearner()

    with pytest.raises(RuntimeError, match="policy_anchor_model_state_dict must be a dict"):
        apply_minimal_checkpoint_resume_state(
            checkpoint_path=tmp_path / "checkpoint.pt",
            learner=learner,
            contract=_contract({"policy_anchor_model_state_dict": ["bad-anchor"]}),
            restore_model_guidance=lambda _model, _payload: None,
            restore_counters=True,
        )

    assert learner.optimizer.loaded_state is None
    assert learner._grad_scaler.loaded_state is None


def test_apply_checkpoint_resume_counters_restores_counters_and_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _ResumeLearner()
    monkeypatch.setattr("weiss_rl.training.checkpointing.restore_state.time.time", lambda: 123.5)

    counters = apply_checkpoint_resume_counters(
        learner=learner,
        payload=_contract().payload,
        restore_counters=True,
    )

    assert counters == checkpoint_counter_state_from_payload(_contract().payload)
    assert learner.update_count == 3
    assert learner.policy_version == 7
    assert learner.total_samples_processed == 42
    assert learner.init_schedule_offset_updates == 5
    assert learner.start_time == 123.5


def test_apply_minimal_checkpoint_initialization_clears_anchor_without_optimizer_or_counters(tmp_path: Path) -> None:
    learner = _InitLearnerWithoutReset()

    source = apply_minimal_checkpoint_initialization(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=_contract(),
        restore_model_guidance=lambda _model, _payload: None,
    )

    assert source.update_count == 3
    assert source.policy_version == 7
    assert source.total_samples_processed == 42
    assert learner.update_count == 99
    assert learner.policy_version == 88
    assert learner.total_samples_processed == 77
    assert learner.model.loaded_state is not None
    assert learner.optimizer.loaded_state is None
    assert learner._grad_scaler.loaded_state is None
    assert learner.loaded_anchor_state is None


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
