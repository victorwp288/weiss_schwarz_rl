from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.training.checkpointing.restore import apply_minimal_checkpoint_resume_state

from .checkpoint_restore_test_support import (
    OrderedResumeLearnerDouble,
    ResumeLearnerDouble,
    minimal_checkpoint_contract,
)


def test_apply_minimal_checkpoint_resume_state_restores_training_state(tmp_path: Path) -> None:
    learner = ResumeLearnerDouble()
    guidance_calls: list[tuple[object, dict[str, object]]] = []

    resume = apply_minimal_checkpoint_resume_state(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=minimal_checkpoint_contract(),
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
    learner = OrderedResumeLearnerDouble()

    resume = apply_minimal_checkpoint_resume_state(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=minimal_checkpoint_contract(),
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
    learner = ResumeLearnerDouble()

    with pytest.raises(RuntimeError, match="policy_anchor_model_state_dict must be a dict"):
        apply_minimal_checkpoint_resume_state(
            checkpoint_path=tmp_path / "checkpoint.pt",
            learner=learner,
            contract=minimal_checkpoint_contract({"policy_anchor_model_state_dict": ["bad-anchor"]}),
            restore_model_guidance=lambda _model, _payload: None,
            restore_counters=True,
        )

    assert learner.optimizer.loaded_state is None
    assert learner._grad_scaler.loaded_state is None
