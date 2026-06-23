from __future__ import annotations

from pathlib import Path

from weiss_rl.training.checkpointing.restore import apply_minimal_checkpoint_initialization

from .checkpoint_restore_test_support import InitLearnerWithoutResetDouble, minimal_checkpoint_contract


def test_apply_minimal_checkpoint_initialization_clears_anchor_without_optimizer_or_counters(tmp_path: Path) -> None:
    learner = InitLearnerWithoutResetDouble()

    source = apply_minimal_checkpoint_initialization(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        contract=minimal_checkpoint_contract(),
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
