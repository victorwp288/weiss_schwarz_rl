from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.loop.setup import (
    effective_init_schedule_offset_from_checkpoint,
    infer_init_schedule_offset_from_scalars,
    publish_initial_runtime_snapshot_after_resume,
)
from weiss_rl.training.loop.update_schedule import schedule_update_count_for_next_update


def test_publish_initial_runtime_snapshot_after_resume_forces_current_update() -> None:
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        calls.append(dict(kwargs))
        return {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}

    runtime.maybe_publish_snapshot = _publish_snapshot
    model = object()

    metrics = publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=model, update_count=25)

    assert metrics == {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}
    assert calls == [{"learner_model": model, "learner_update_count": 25, "force": True}]


def test_publish_initial_runtime_snapshot_after_resume_skips_zero_update() -> None:
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        raise AssertionError("zero-update fresh runs must not publish a resume snapshot")

    runtime.maybe_publish_snapshot = _publish_snapshot

    metrics = publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=object(), update_count=0)

    assert metrics == {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}


def test_schedule_update_count_for_next_update_preserves_source_checkpoint_time() -> None:
    assert (
        schedule_update_count_for_next_update(
            learner_update_count=0,
            init_schedule_offset_updates=90,
        )
        == 91
    )
    assert (
        schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_preserves_nested_warmstart_time() -> None:
    assert (
        effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_allows_explicit_override() -> None:
    assert (
        effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
            override_updates=0,
        )
        == 0
    )
    assert (
        effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
            override_updates=12,
        )
        == 12
    )


def test_infer_init_schedule_offset_from_scalars_recovers_latest_offset(tmp_path: Path) -> None:
    scalars_path = tmp_path / "scalars.jsonl"
    scalars_path.write_text(
        "\n".join(
            (
                '{"update_count": 1, "init_schedule_offset_updates": 90}',
                '{"update_count": 2}',
                '{"update_count": 3, "init_schedule_offset_updates": 90.0}',
            )
        ),
        encoding="utf-8",
    )

    assert infer_init_schedule_offset_from_scalars(scalars_path) == 90
