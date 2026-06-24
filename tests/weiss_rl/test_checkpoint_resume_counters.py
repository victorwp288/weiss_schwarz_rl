from __future__ import annotations

import pytest
from weiss_rl.training.checkpointing.storage.restore_state import (
    apply_checkpoint_resume_counters,
    checkpoint_counter_state_from_payload,
)

from .checkpoint_restore_test_support import ResumeLearnerDouble, minimal_checkpoint_contract


def test_apply_checkpoint_resume_counters_restores_counters_and_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = ResumeLearnerDouble()
    monkeypatch.setattr("weiss_rl.training.checkpointing.storage.restore_state.time.time", lambda: 123.5)

    counters = apply_checkpoint_resume_counters(
        learner=learner,
        payload=minimal_checkpoint_contract().payload,
        restore_counters=True,
    )

    assert counters == checkpoint_counter_state_from_payload(minimal_checkpoint_contract().payload)
    assert learner.update_count == 3
    assert learner.policy_version == 7
    assert learner.total_samples_processed == 42
    assert learner.init_schedule_offset_updates == 5
    assert learner.start_time == 123.5
