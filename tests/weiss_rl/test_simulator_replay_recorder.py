from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.replay.simulator_replay import SimulatorReplayRecorder
from weiss_rl.eval.simulator.harness import ScheduledGame

from .eval_runner_test_support import make_decision_batch, make_scheduled_game


class _RecordingReplayPool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def enable_replay_sampling(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))


def _stack_with_visibility(visibility: str) -> Any:
    return SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace(observation_visibility=visibility)))


def test_simulator_replay_recorder_starts_capture_and_records_steps(tmp_path) -> None:
    layout = ArtifactLayout.from_run_dir(tmp_path)
    layout.ensure_directories()
    pool = _RecordingReplayPool()
    recorder = SimulatorReplayRecorder(
        stack=cast(Any, _stack_with_visibility("public")),
        artifact_layout=layout,
        run_id256_bytes=bytes.fromhex("ab" * 32),
        spec_hash256_bytes=bytes.fromhex("cd" * 32),
        capture_rate=1.0,
        capture_limit=1,
    )
    scheduled_game = make_scheduled_game(ScheduledGame, focal_policy_id="candidate")

    capture = recorder.maybe_start(env=cast(Any, SimpleNamespace(pool=pool)), scheduled_game=scheduled_game)

    assert capture is not None
    assert capture.raw_dir.exists()
    assert pool.calls == [
        {
            "sample_rate": 1.0,
            "out_dir": capture.raw_dir.as_posix(),
            "compress": False,
            "visibility_mode": "public",
            "store_actions": True,
        }
    ]

    recorder.record_initial_batch(capture, batch=make_decision_batch(terminal=False))
    recorder.record_step(
        capture,
        decision_id=3,
        actor=0,
        action=0,
        next_batch=make_decision_batch(terminal=True, decision_id=4),
        legal_ids=np.array([0], dtype=np.uint32),
    )

    assert capture.simulator_episode_key == 1
    assert capture.steps is not None
    assert len(capture.steps) == 1
    assert capture.steps[0].decision_id == 3
    assert capture.steps[0].actor == 0
    assert capture.steps[0].action == 0

    assert recorder.maybe_start(env=cast(Any, SimpleNamespace(pool=pool)), scheduled_game=scheduled_game) is None
