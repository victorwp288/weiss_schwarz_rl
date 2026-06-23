from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from weiss_rl.training.warmstart import run_structured_warmstart


class _Learner:
    def __init__(self) -> None:
        self.update_count = 0
        self.coef_calls: list[dict[str, Any]] = []
        self.auxiliary_batches: list[Any] = []

    def set_teacher_aux_coefs(self, **kwargs: Any) -> None:
        self.coef_calls.append(kwargs)

    def auxiliary_update(self, batch: Any) -> dict[str, float]:
        self.auxiliary_batches.append(batch)
        self.update_count += 1
        return {"learner_loss": float(self.update_count)}

    def get_policy_version(self) -> int:
        return 100 + int(self.update_count)


class _TensorBoardLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log_training_step(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class _Runtime:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def structured_warmstart_source_mix(self) -> Iterator[dict[str, float]]:
        self.events.append("source-enter")
        try:
            yield {"structured_warmstart_source_count": 2.0}
        finally:
            self.events.append("source-exit")

    @contextmanager
    def disable_mirror_policy_fusion(self) -> Iterator[None]:
        self.events.append("fusion-enter")
        try:
            yield
        finally:
            self.events.append("fusion-exit")


def _training_config(*, enabled: bool = True, updates: int = 2) -> SimpleNamespace:
    warmstart = SimpleNamespace(
        updates=updates,
        teacher_family_coef=0.75,
        teacher_slot_coef=0.35,
        teacher_hand_coef=0.45,
        teacher_move_source_coef=0.25,
        teacher_attack_type_coef=0.2,
        teacher_action_coef=0.5,
        teacher_same_family_action_coef=0.15,
        teacher_public_heuristic_coef=0.8,
        teacher_public_heuristic_temperature=12.0,
        teacher_public_heuristic_families=("main_play_character",),
        teacher_public_heuristic_profiles=("base", "aggressive"),
        teacher_public_heuristic_profile_mode="mixture",
        teacher_public_heuristic_profiles_end_updates=4,
    )
    return SimpleNamespace(
        structured_warmstart_enabled=enabled,
        structured_warmstart=warmstart,
        teacher_family_coef=0.1,
        teacher_slot_coef=0.2,
        teacher_hand_coef=0.25,
        teacher_move_source_coef=0.3,
        teacher_attack_type_coef=0.4,
        teacher_action_coef=0.5,
        teacher_same_family_action_coef=0.6,
        teacher_public_heuristic_coef=0.7,
        teacher_public_heuristic_temperature=1.5,
        teacher_public_heuristic_families=("play",),
        teacher_public_heuristic_profiles=("base",),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=9,
    )


def test_run_structured_warmstart_collects_updates_logs_and_restores_coefficients(tmp_path: Path) -> None:
    learner = _Learner()
    runtime = _Runtime()
    tensorboard = _TensorBoardLogger()
    scalars_records: list[dict[str, Any]] = []
    collected: list[dict[str, Any]] = []

    def collect_training_batch_fn(**kwargs: Any) -> SimpleNamespace:
        collected.append(kwargs)
        return SimpleNamespace(
            learner_batch=f"batch-{len(collected)}",
            runtime_metrics={"runtime_envs": float(len(collected))},
        )

    def write_scalars_record_fn(**kwargs: Any) -> None:
        scalars_records.append(kwargs)

    latest = run_structured_warmstart(
        learner=learner,
        runtime=runtime,
        algorithm="impala_vtrace_structured_v1",
        training_config=_training_config(),
        rewards_config=SimpleNamespace(name="rewards"),
        training_paths=SimpleNamespace(scalars_path=tmp_path / "scalars.jsonl"),
        tensorboard_logger=tensorboard,
        start_time=10.0,
        collect_training_batch_fn=collect_training_batch_fn,
        write_scalars_record_fn=write_scalars_record_fn,
        time_fn=lambda: 15.5,
    )

    assert learner.auxiliary_batches == ["batch-1", "batch-2"]
    assert [record["metrics"]["warmstart_step"] for record in scalars_records] == [1.0, 2.0]
    assert [record["metrics"]["runtime_envs"] for record in scalars_records] == [1.0, 2.0]
    assert [record["metrics"]["structured_warmstart_source_count"] for record in scalars_records] == [2.0, 2.0]
    assert [record["wall_clock_seconds"] for record in tensorboard.records] == [5.5, 5.5]
    assert [record["policy_version"] for record in tensorboard.records] == [101, 102]
    assert latest["learner_loss"] == 2.0
    assert latest["warmstart_phase"] == 1.0
    assert latest["warmstart_step"] == 2.0
    assert runtime.events == ["source-enter", "fusion-enter", "fusion-exit", "source-exit"]

    assert learner.coef_calls[0]["family"] == 0.75
    assert learner.coef_calls[0]["hand"] == 0.45
    assert learner.coef_calls[0]["public_heuristic_profiles"] == ("base", "aggressive")
    assert learner.coef_calls[-1]["family"] == 0.1
    assert learner.coef_calls[-1]["hand"] == 0.25
    assert learner.coef_calls[-1]["public_heuristic_profiles"] == ("base",)


def test_run_structured_warmstart_skips_disabled_and_zero_update_configs() -> None:
    learner = _Learner()
    runtime = _Runtime()

    assert (
        run_structured_warmstart(
            learner=learner,
            runtime=runtime,
            algorithm="impala_vtrace_structured_v1",
            training_config=_training_config(enabled=False),
            rewards_config=SimpleNamespace(),
            training_paths=SimpleNamespace(scalars_path=Path("unused")),
            tensorboard_logger=None,
            start_time=0.0,
        )
        == {}
    )
    assert (
        run_structured_warmstart(
            learner=learner,
            runtime=runtime,
            algorithm="impala_vtrace_structured_v1",
            training_config=_training_config(updates=0),
            rewards_config=SimpleNamespace(),
            training_paths=SimpleNamespace(scalars_path=Path("unused")),
            tensorboard_logger=None,
            start_time=0.0,
        )
        == {}
    )
    assert learner.coef_calls == []
    assert runtime.events == []


def test_run_structured_warmstart_rejects_non_impala_algorithm() -> None:
    with pytest.raises(RuntimeError, match="structured warmstart currently supports only IMPALA learners"):
        run_structured_warmstart(
            learner=_Learner(),
            runtime=_Runtime(),
            algorithm="ppo_lite_masked_v1",
            training_config=_training_config(),
            rewards_config=SimpleNamespace(),
            training_paths=SimpleNamespace(scalars_path=Path("unused")),
            tensorboard_logger=None,
            start_time=0.0,
        )
