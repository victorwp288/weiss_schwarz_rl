from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from weiss_rl.training.checkpoints import (
    OBSERVED_BEST_CHECKPOINT_FILENAME,
    append_checkpoint_guard_event,
    best_checkpoint_record,
    checkpoint_path_for_update,
    current_focal_policy_id,
    ensure_current_checkpoint,
    extract_structured_guard_b2_anchor_score,
    initialize_model_from_checkpoint,
    maybe_log_structured_mainmove_guard,
    publish_checkpoint_aliases,
    restore_minimal_train_checkpoint,
    write_minimal_train_checkpoint,
    write_scalars_record,
)


class _Model:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Optimizer:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"lr": 0.01}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _GradScaler:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"scale": 2.0}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Learner:
    update_count = 3
    total_samples_processed = 42

    def __init__(self, *, model=...) -> None:
        self.model = _Model() if model is ... else model
        self.optimizer = _Optimizer()
        self._grad_scaler = _GradScaler()
        self.policy_version = 7
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.anchor_state: dict[str, torch.Tensor] | None = None
        self.loaded_anchor_state: dict[str, torch.Tensor] | None = None
        self.reset_anchor_calls = 0

    def get_policy_version(self) -> int:
        return self.policy_version

    def _optimizer_for_step(self) -> _Optimizer:
        return self.optimizer

    def policy_anchor_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self.anchor_state

    def load_policy_anchor_state_dict(self, state_dict) -> None:
        self.loaded_anchor_state = state_dict

    def reset_policy_anchor_to_current_model(self) -> None:
        self.reset_anchor_calls += 1


class _TrainingPaths:
    def __init__(self, root: Path) -> None:
        self.checkpoints_dir = root
        self.checkpoint_tracker_path = root / "checkpoint_tracker.json"
        self.logs_dir = root / "logs"
        self.latest_checkpoint_path = root / "latest.pt"
        self.best_checkpoint_path = root / "best.pt"
        self.snapshots_dir = root / "snapshots"


def test_write_scalars_record_appends_stable_json_line(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.training.checkpoints.time.time", lambda: 105.25)
    scalars_path = tmp_path / "scalars.jsonl"

    record = write_scalars_record(
        scalars_path=scalars_path,
        learner=_Learner(),
        metrics={"loss": 1.5},
        start_time=100.0,
    )

    assert record["update_count"] == 3
    assert record["policy_version"] == 7
    assert record["wall_clock_seconds"] == pytest.approx(5.25)
    assert record["wall_clock_ms"] == 5250
    assert json.loads(scalars_path.read_text(encoding="utf-8")) == record


def test_ensure_current_checkpoint_reuses_existing_file_and_writes_missing_file(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoints_dir.mkdir(parents=True)
    learner = _Learner()
    existing_path = checkpoint_path_for_update(paths.checkpoints_dir, update_count=learner.update_count)
    existing_path.write_bytes(b"existing")
    write_calls: list[Path] = []

    assert current_focal_policy_id(learner=learner) == "train_u3_p7"
    assert (
        ensure_current_checkpoint(
            training_paths=paths,
            learner=learner,
            write_checkpoint=lambda path: write_calls.append(path),
        )
        == existing_path
    )
    assert write_calls == []

    learner.update_count = 4

    def _write_checkpoint(path: Path) -> None:
        write_calls.append(path)
        path.write_bytes(b"new")

    new_path = ensure_current_checkpoint(
        training_paths=paths,
        learner=learner,
        write_checkpoint=_write_checkpoint,
    )

    assert new_path == paths.checkpoints_dir / "checkpoint_4.pt"
    assert new_path.read_bytes() == b"new"
    assert write_calls == [new_path]


def test_write_minimal_train_checkpoint_payload_shape_and_save(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    learner = _Learner()
    learner.init_schedule_offset_updates = 90

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )

    assert checkpoint_path.is_file()
    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 3
    assert payload["policy_version"] == 7
    assert payload["device"] == "cpu"
    assert payload["config_hash256"] == "abc"
    assert payload["spec_hash256"] == "def"
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert payload["recurrent_core"] == "gru"
    assert payload["total_samples_processed"] == 42
    assert payload["init_schedule_offset_updates"] == 90
    assert payload["policy_anchor_model_state_dict"] is None
    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.25)
    assert payload["optimizer_state_dict"] == {"lr": 0.01}
    assert payload["grad_scaler_state_dict"] == {"scale": 2.0}
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert loaded["update_count"] == payload["update_count"]
    assert loaded["model_state_dict"]["weight"].tolist() == [1.0]


def test_minimal_train_checkpoint_round_trips_policy_anchor_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.anchor_state = {"anchor_weight": torch.tensor([2.0])}

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
    )
    restored = _Learner()
    restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda _model, _payload: None,
    )

    assert payload["policy_anchor_model_state_dict"]["anchor_weight"].tolist() == [2.0]
    assert restored.loaded_anchor_state is not None
    assert restored.loaded_anchor_state["anchor_weight"].tolist() == [2.0]


def test_write_minimal_train_checkpoint_requires_model(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Cannot write a checkpoint without a learner model"):
        write_minimal_train_checkpoint(
            checkpoint_path=tmp_path / "checkpoint.pt",
            learner=_Learner(model=None),
            device=torch.device("cpu"),
            config_hash256="abc",
        )


def test_restore_minimal_train_checkpoint_restores_state_and_can_preserve_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )
    restored = _Learner()
    restored.update_count = 99
    restored.policy_version = 88
    restored.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    resume = restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
        restore_counters=False,
    )

    assert resume.update_count == 99
    assert resume.policy_version == 88
    assert resume.total_samples_processed == 77
    assert restored.model.loaded_state is not None
    assert restored.optimizer.loaded_state == {"lr": 0.01}
    assert restored.init_schedule_offset_updates == 90
    assert resume.init_schedule_offset_updates == 90
    assert guidance_calls[0][0] is restored.model
    assert guidance_calls[0][1] == pytest.approx(0.25)


def test_initialize_model_from_checkpoint_loads_weights_without_optimizer_or_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="source-config",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.5},
    )
    initialized = _Learner()
    initialized.update_count = 99
    initialized.policy_version = 88
    initialized.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    source = initialize_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=initialized,
        device=torch.device("cpu"),
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
    )

    assert source.update_count == 3
    assert source.policy_version == 7
    assert source.total_samples_processed == 42
    assert source.init_schedule_offset_updates == 90
    assert initialized.update_count == 99
    assert initialized.policy_version == 88
    assert initialized.total_samples_processed == 77
    assert initialized.model.loaded_state is not None
    assert initialized.optimizer.loaded_state is None
    assert initialized._grad_scaler.loaded_state is None
    assert initialized.loaded_anchor_state is None
    assert initialized.reset_anchor_calls == 1
    assert guidance_calls[0][0] is initialized.model
    assert guidance_calls[0][1] == pytest.approx(0.5)


def test_publish_checkpoint_aliases_updates_latest_and_promotes_lower_training_loss(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    stack = SimpleNamespace(config=SimpleNamespace(evaluation=None))
    learner = _Learner()
    run_dir = tmp_path

    checkpoint_a = tmp_path / "checkpoint_a.pt"
    checkpoint_a.write_bytes(b"checkpoint-a")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_a,
        learner=learner,
        latest_metrics={"loss": 1.5},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert tracker["latest"]["metric_kind"] == "training_loss"
    assert tracker["latest"]["metric_value"] == pytest.approx(1.5)
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_a.pt"

    learner.update_count = 4
    checkpoint_b = tmp_path / "checkpoint_b.pt"
    checkpoint_b.write_bytes(b"checkpoint-b")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_b,
        learner=learner,
        latest_metrics={"loss": 2.0},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-b"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert tracker["latest"]["source_checkpoint_path"] == "checkpoint_b.pt"
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_a.pt"

    learner.update_count = 5
    checkpoint_c = tmp_path / "checkpoint_c.pt"
    checkpoint_c.write_bytes(b"checkpoint-c")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_c,
        learner=learner,
        latest_metrics={"loss": 1.0},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-c"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-c"
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_c.pt"
    assert json.loads(paths.checkpoint_tracker_path.read_text(encoding="utf-8")) == tracker
    assert best_checkpoint_record(paths) == tracker["best"]


def test_publish_checkpoint_aliases_records_dev_eval_ineligibility_on_latest(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    stack = SimpleNamespace(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(periodic_dev_eval_interval_updates=25),
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(enabled=False, truncation_rate_threshold=0.25),
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    promote_min_prob_gt_half=0.60,
                    promote_max_ci_half_width=0.24,
                ),
            ),
        )
    )
    learner = _Learner()
    checkpoint = tmp_path / "checkpoint_25.pt"
    checkpoint.write_bytes(b"checkpoint")

    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={
            "aggregate_score": 0.62,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.58, "prob_lt_half": 0.42, "ci_half_width": 0.12},
                }
            },
        },
    )

    assert tracker["latest"]["metric_kind"] is None
    assert tracker["best"] is None
    assert tracker["observed_best"]["alias"] == "observed_best"
    assert tracker["observed_best"]["metric_kind"] == "dev_eval_observed_mean"
    assert tracker["observed_best"]["metric_value"] == pytest.approx(0.62)
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    observed_best_path = paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME
    assert observed_best_path.read_bytes() == b"checkpoint"
    candidate = tracker["latest"]["dev_eval_candidate"]
    assert candidate["score"] == pytest.approx(0.62)
    assert candidate["eligible_for_best"] is False
    assert candidate["ineligibility_reasons"] == ["confidence_prob"]
    assert candidate["confidence"]["min_prob_gt_half"] == pytest.approx(0.58)

    learner.update_count = 50
    lower_checkpoint = tmp_path / "checkpoint_50.pt"
    lower_checkpoint.write_bytes(b"lower-checkpoint")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=lower_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.25},
        dev_eval_summary={
            "aggregate_score": 0.59,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.57, "prob_lt_half": 0.43, "ci_half_width": 0.12},
                }
            },
        },
    )

    assert tracker["latest"]["source_checkpoint_path"] == "checkpoint_50.pt"
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    assert observed_best_path.read_bytes() == b"checkpoint"


def test_append_checkpoint_guard_event_writes_sorted_jsonl(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")

    append_checkpoint_guard_event(paths, {"z": 2, "a": 1})
    append_checkpoint_guard_event(paths, {"event": "second"})

    event_path = paths.logs_dir / "checkpoint_guard.jsonl"
    assert event_path.read_text(encoding="utf-8").splitlines() == [
        '{"a": 1, "z": 2}',
        '{"event": "second"}',
    ]


def test_maybe_log_structured_mainmove_guard_writes_warning_when_learning_is_weak(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    learner = _Learner()

    payload = maybe_log_structured_mainmove_guard(
        training_paths=paths,
        learner=learner,
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.4,
            "structured_main_move_share_when_play_available": 0.5,
        },
        dev_eval_summary={
            "aggregate_score": 0.2,
            "anchor_scores": {"B2 HeuristicPublic": 0.0},
        },
    )

    assert payload is not None
    assert payload["event_kind"] == "structured_mainmove_warning_v1"
    assert payload["b2_anchor_score"] == pytest.approx(0.0)
    assert payload["dev_eval_aggregate_score"] == pytest.approx(0.2)
    log_payload = json.loads((paths.logs_dir / "checkpoint_guard.jsonl").read_text(encoding="utf-8"))
    assert log_payload == payload
    assert extract_structured_guard_b2_anchor_score({"anchor_scores": {"B2": 0.125}}) == pytest.approx(0.125)


def test_maybe_log_structured_mainmove_guard_suppresses_when_b2_score_is_healthy(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")

    payload = maybe_log_structured_mainmove_guard(
        training_paths=paths,
        learner=_Learner(),
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.4,
            "structured_main_move_share_when_play_available": 0.5,
        },
        dev_eval_summary={
            "aggregate_score": 0.2,
            "anchor_scores": {"B2 HeuristicPublic": 0.11},
        },
    )

    assert payload is None
    assert not (paths.logs_dir / "checkpoint_guard.jsonl").exists()
