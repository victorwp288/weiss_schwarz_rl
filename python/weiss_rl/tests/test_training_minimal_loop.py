from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.minimal_loop import (
    _POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    _effective_init_schedule_offset_from_checkpoint,
    _infer_init_schedule_offset_from_scalars,
    _merge_post_update_auxiliary_metrics_into_training_log,
    _publish_initial_runtime_snapshot_after_resume,
    _schedule_update_count_for_next_update,
)
from weiss_rl.training.train_entrypoint_main import _require_explicit_resume_geometry


def test_publish_initial_runtime_snapshot_after_resume_forces_current_update() -> None:
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        calls.append(dict(kwargs))
        return {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}

    runtime.maybe_publish_snapshot = _publish_snapshot
    model = object()

    metrics = _publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=model, update_count=25)

    assert metrics == {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}
    assert calls == [{"learner_model": model, "learner_update_count": 25, "force": True}]


def test_publish_initial_runtime_snapshot_after_resume_skips_zero_update() -> None:
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        raise AssertionError("zero-update fresh runs must not publish a resume snapshot")

    runtime.maybe_publish_snapshot = _publish_snapshot

    metrics = _publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=object(), update_count=0)

    assert metrics == {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}


def test_schedule_update_count_for_next_update_preserves_source_checkpoint_time() -> None:
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=0,
            init_schedule_offset_updates=90,
        )
        == 91
    )
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_preserves_nested_warmstart_time() -> None:
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_allows_explicit_override() -> None:
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
            override_updates=0,
        )
        == 0
    )
    assert (
        _effective_init_schedule_offset_from_checkpoint(
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

    assert _infer_init_schedule_offset_from_scalars(scalars_path) == 90


def test_merge_post_update_auxiliary_metrics_into_training_log_uses_latest_learner_record() -> None:
    calls: list[dict[str, object]] = []
    logger = SimpleNamespace(
        merge_latest_custom_metrics=lambda **kwargs: calls.append(dict(kwargs)),
    )
    learner = SimpleNamespace(
        logger=logger,
        update_count=7,
        get_policy_version=lambda: 3,
    )
    metrics = {"paired_swing_replay_loss": 0.25}

    _merge_post_update_auxiliary_metrics_into_training_log(learner=learner, metrics=metrics)

    assert calls == [
        {
            "update_count": 7,
            "policy_version": 3,
            "metrics": metrics,
            "prefixes": _POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
        }
    ]
    assert "pfsp_" in calls[0]["prefixes"]
    assert "collector_pfsp_" in calls[0]["prefixes"]


def test_resume_requires_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=None,
        unroll_length=None,
        runtime_mode=None,
        profile=None,
    )

    try:
        _require_explicit_resume_geometry(parser, args)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("resume without explicit geometry should fail")

    assert "--num-envs" in message
    assert "--unroll-length" in message
    assert "--runtime-mode" in message
    assert "--profile" in message


def test_resume_accepts_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=32,
        unroll_length=16,
        runtime_mode="train_async_fast",
        profile="fast",
    )

    _require_explicit_resume_geometry(parser, args)
