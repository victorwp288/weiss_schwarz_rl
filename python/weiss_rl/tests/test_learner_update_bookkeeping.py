from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.update_bookkeeping import (
    learner_acceleration_state,
    record_timing_ms,
    should_emit_structured_metrics,
    teacher_aux_active,
    throughput_metrics,
)


def test_record_timing_ms_accumulates_only_when_enabled() -> None:
    metrics: dict[str, float] = {}

    record_timing_ms(metrics, profile_timers=False, name="loss", elapsed_seconds=0.25)
    assert metrics == {}

    record_timing_ms(None, profile_timers=True, name="loss", elapsed_seconds=0.25)
    assert metrics == {}

    record_timing_ms(metrics, profile_timers=True, name="loss", elapsed_seconds=0.25)
    record_timing_ms(metrics, profile_timers=True, name="loss", elapsed_seconds=0.125)

    assert metrics == {"timer_loss_ms": 375.0}


@pytest.mark.parametrize(
    ("mode", "auxiliary_update", "expected"),
    [
        ("off", False, False),
        ("off", True, False),
        ("warmstart_only", False, False),
        ("warmstart_only", True, True),
        ("always", False, True),
        ("always", True, True),
    ],
)
def test_teacher_aux_active_preserves_mode_rules(mode: str, auxiliary_update: bool, expected: bool) -> None:
    assert teacher_aux_active(teacher_aux_mode=mode, auxiliary_update=auxiliary_update) is expected


@pytest.mark.parametrize(
    ("mode", "auxiliary_update", "update_count", "expected"),
    [
        ("off", False, 10, False),
        ("sampled", True, 10, False),
        ("sampled", False, 9, False),
        ("sampled", False, 10, True),
        ("full", True, 9, True),
        ("full", False, 9, True),
    ],
)
def test_should_emit_structured_metrics_preserves_mode_and_sampling_rules(
    mode: str,
    auxiliary_update: bool,
    update_count: int,
    expected: bool,
) -> None:
    assert (
        should_emit_structured_metrics(
            structured_metrics_mode=mode,
            auxiliary_update=auxiliary_update,
            update_count=update_count,
        )
        is expected
    )


def test_throughput_metrics_uses_elapsed_floor() -> None:
    assert throughput_metrics(total_samples_processed=20, update_count=4, elapsed_seconds=2.0) == (10.0, 2.0)
    assert throughput_metrics(total_samples_processed=20, update_count=4, elapsed_seconds=0.0) == (
        20_000_000.0,
        4_000_000.0,
    )


def test_learner_acceleration_state_disables_amp_without_cuda_model() -> None:
    assert learner_acceleration_state(model=None, mixed_precision=True) == (False, "cpu", None)
    assert learner_acceleration_state(model=torch.nn.Identity(), mixed_precision=True) == (False, "cpu", None)

    amp_enabled, device_type, grad_scaler = learner_acceleration_state(
        model=torch.nn.Linear(2, 1),
        mixed_precision=True,
    )

    assert amp_enabled is False
    assert device_type == "cpu"
    assert grad_scaler is None
