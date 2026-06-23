from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.eval.simulator_runner import ResolvedEvalPolicy, SimulatorEvalRunner

from .eval_runner_test_support import (
    FakeEvalEnv,
    RecordingEvalModel,
    make_decision_batch,
    make_scheduled_game,
)


def test_simulator_eval_runner_uses_learner_scoring_mode(tmp_path: Path, monkeypatch) -> None:
    env = FakeEvalEnv(
        reset_batch=make_decision_batch(terminal=False),
        step_batch=make_decision_batch(terminal=True, decision_id=1),
    )
    model = RecordingEvalModel()
    layout = ArtifactLayout.from_run_dir(tmp_path)
    layout.ensure_directories()
    runner = SimulatorEvalRunner(
        stack=cast(Any, SimpleNamespace(config=SimpleNamespace(curriculum=None))),
        policies={
            "candidate": ResolvedEvalPolicy(
                policy_id="candidate",
                kind="snapshot_registry",
                model=cast(Any, model),
            )
        },
        artifact_layout=layout,
        run_id256="ab" * 32,
        spec_hash256="cd" * 32,
        action_dim=1,
        pass_action_id=0,
        require_sorted_legal_ids=False,
        replay_capture_rate=0.0,
        regression_capture_count=0,
    )
    monkeypatch.setattr(runner, "_build_ids_eval_env", lambda *, seed, scheduled_game=None: env)
    scheduled_game = make_scheduled_game(
        ScheduledGame,
        focal_policy_id="candidate",
        seat0_policy_id="candidate",
        seat1_policy_id="candidate",
    )

    result = runner.run_game(scheduled_game)

    assert result.episode_seed == scheduled_game.episode_seed
    assert env.reset_seed == scheduled_game.episode_seed
    assert model.scoring_modes == ["learner"]
    assert env.closed is True
