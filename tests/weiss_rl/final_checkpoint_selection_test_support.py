from __future__ import annotations

from pathlib import Path

from weiss_rl.training.checkpointing.lifecycle.finalization import TrainingFinalCheckpointHooks


class RecordingTensorBoardLogger:
    def __init__(self, events: list[tuple[str, dict[str, object]]]) -> None:
        self._events = events

    def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
        self._events.append(("tensorboard", {"payload": payload, "step": step}))


class CheckpointHookRecorder:
    def __init__(
        self,
        *,
        events: list[tuple[str, dict[str, object]]],
        checkpoint_path: Path,
        alias_payload: dict[str, object],
        guard_event: dict[str, object] | None = None,
        loaded_tracker: dict[str, object] | None = None,
        fail_on_load: bool = False,
    ) -> None:
        self.events = events
        self.checkpoint_path = checkpoint_path
        self.alias_payload = alias_payload
        self.guard_event = guard_event
        self.loaded_tracker = {} if loaded_tracker is None else loaded_tracker
        self.fail_on_load = fail_on_load

    def hooks(self) -> TrainingFinalCheckpointHooks:
        return TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=self.ensure_current_checkpoint,
            publish_checkpoint_aliases=self.publish_checkpoint_aliases,
            maybe_finalize_from_best_checkpoint=self.maybe_finalize_from_best_checkpoint,
            load_checkpoint_tracker=self.load_checkpoint_tracker,
        )

    def ensure_current_checkpoint(self, **kwargs: object) -> Path:
        self.events.append(("ensure", kwargs))
        return self.checkpoint_path

    def publish_checkpoint_aliases(self, **kwargs: object) -> dict[str, object]:
        self.events.append(("aliases", kwargs))
        return self.alias_payload

    def maybe_finalize_from_best_checkpoint(self, **kwargs: object) -> dict[str, object] | None:
        self.events.append(("finalize", kwargs))
        return self.guard_event

    def load_checkpoint_tracker(self, paths: object) -> dict[str, object]:
        if self.fail_on_load:
            raise AssertionError("tracker should not reload when final guard does not change selection")
        self.events.append(("load_tracker", {"paths": paths}))
        return self.loaded_tracker


def guard_event(
    *,
    update_count: int,
    best_update_count: int,
    current_score: float,
    best_score: float,
) -> dict[str, object]:
    return {
        "update_count": update_count,
        "best_update_count": best_update_count,
        "current_score": current_score,
        "best_score": best_score,
    }
