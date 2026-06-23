from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.minimal.promotion import TrainingCheckpointPromotionHooks, _maybe_checkpoint_and_promote_snapshot


class RecordingModel:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls

    def state_dict(self) -> dict[str, object]:
        self._calls.append(("state_dict", {}))
        return {"weight": 1}


class RecordingRuntime:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls

    def refresh_opponent_pool(self) -> None:
        self._calls.append(("refresh", {}))


class RecordingTensorBoardLogger:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls

    def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
        self._calls.append(("tensorboard", {"payload": payload, "step": step}))


def make_learner(
    calls: list[tuple[str, dict[str, object]]],
    *,
    update_count: int = 6,
    model: object | None = None,
    policy_version: int = 11,
) -> SimpleNamespace:
    resolved_model = RecordingModel(calls) if model is None else model
    return SimpleNamespace(update_count=update_count, model=resolved_model, get_policy_version=lambda: policy_version)


def recording_hooks(
    calls: list[tuple[str, dict[str, object]]],
    *,
    tracker_payload: dict[str, object] | None = None,
    candidate_policy_id: str = "candidate_policy",
    promotion_passed: bool = True,
) -> TrainingCheckpointPromotionHooks:
    def write_checkpoint(**kwargs: object) -> None:
        calls.append(("write", kwargs))

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        calls.append(("aliases", kwargs))
        return tracker_payload or {"latest": {"update": 6}}

    def maybe_log_structured_mainmove_guard(**kwargs: object) -> None:
        calls.append(("guard", kwargs))

    def persist_snapshot_registry_entry(**kwargs: object) -> str:
        calls.append(("persist", kwargs))
        return candidate_policy_id

    def run_snapshot_promotion_gate(**kwargs: object) -> bool:
        calls.append(("promotion", kwargs))
        return promotion_passed

    return TrainingCheckpointPromotionHooks(
        write_checkpoint=write_checkpoint,
        publish_checkpoint_aliases=publish_checkpoint_aliases,
        maybe_log_structured_mainmove_guard=maybe_log_structured_mainmove_guard,
        persist_snapshot_registry_entry=persist_snapshot_registry_entry,
        run_snapshot_promotion_gate=run_snapshot_promotion_gate,
    )


def run_checkpoint_promotion(
    *,
    tmp_path: Path,
    learner: object,
    runtime: object,
    hooks: TrainingCheckpointPromotionHooks,
    latest_metrics: dict[str, object] | None = None,
    last_dev_eval_summary: dict[str, object] | None = None,
    checkpoint_interval_updates: int = 3,
    tensorboard_logger: object | None = None,
    stack: object | None = None,
    contract: object | None = None,
    device: object | None = None,
    algorithm: object | None = None,
) -> object:
    return _maybe_checkpoint_and_promote_snapshot(
        learner=learner,
        stack=stack or object(),
        contract=contract or object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
        runtime=runtime,
        device=device or object(),
        spec_hash256="spec-hash",
        algorithm=algorithm or object(),
        latest_metrics=latest_metrics or {"loss": 1.0},
        last_dev_eval_summary=last_dev_eval_summary,
        checkpoint_interval_updates=checkpoint_interval_updates,
        run_id256="run-id",
        config_hash256="config-hash",
        tensorboard_logger=tensorboard_logger,
        hooks=hooks,
    )
