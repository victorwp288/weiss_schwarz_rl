from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from .training_checkpoint_promotion_test_support import (
    RecordingTensorBoardLogger,
    recording_hooks,
    run_checkpoint_promotion,
)


def test_checkpoint_promotion_missing_model_fails_after_tracker_logging(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    tracker_payload = {"latest": {"update": 6}}

    with pytest.raises(RuntimeError, match="without a learner model"):
        run_checkpoint_promotion(
            tmp_path=tmp_path,
            learner=SimpleNamespace(update_count=6, model=None, get_policy_version=lambda: 11),
            runtime=SimpleNamespace(refresh_opponent_pool=lambda: calls.append(("refresh", {}))),
            tensorboard_logger=RecordingTensorBoardLogger(calls),
            hooks=recording_hooks(calls, tracker_payload=tracker_payload),
        )

    assert [call[0] for call in calls] == ["write", "aliases", "guard", "tensorboard"]
