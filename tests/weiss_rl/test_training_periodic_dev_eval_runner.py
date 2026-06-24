from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .eval_runner_test_support import (
    FakeEvalEnv,
    RecordingEvalModel,
    make_decision_batch,
    make_scheduled_game,
)
from .snapshot_registry_test_support import _load_train_script_module


def test_periodic_dev_eval_runner_resets_env_with_scheduled_episode_seed(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    terminal_batch = make_decision_batch(train_script.DecisionBoundaryBatch, terminal=True)
    env = FakeEvalEnv(reset_batch=terminal_batch)
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PeriodicDevEvalRunner(
        stack=SimpleNamespace(),
        model=RecordingEvalModel(),
        opponent_policy_id="baseline",
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="focal",
        require_sorted_legal_ids=False,
    )
    scheduled_game = make_scheduled_game(train_script.ScheduledGame, focal_policy_id="focal")

    result = runner.run_game(scheduled_game)

    assert env.reset_seed == scheduled_game.episode_seed
    assert env.closed is True
    assert result.episode_seed == scheduled_game.episode_seed


def test_periodic_dev_eval_runner_uses_learner_scoring_mode(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    env = FakeEvalEnv(
        reset_batch=make_decision_batch(train_script.DecisionBoundaryBatch, terminal=False),
        step_batch=make_decision_batch(train_script.DecisionBoundaryBatch, terminal=True, decision_id=1),
    )
    model = RecordingEvalModel()
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PeriodicDevEvalRunner(
        stack=SimpleNamespace(),
        model=model,
        opponent_policy_id="baseline",
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="focal",
        require_sorted_legal_ids=False,
    )
    scheduled_game = make_scheduled_game(train_script.ScheduledGame, focal_policy_id="focal")

    result = runner.run_game(scheduled_game)

    assert model.scoring_modes == ["learner"]
    assert result.total_actions == 1
    assert result.pass_actions == 1
    assert result.pass_with_nonpass_available == 0
    assert env.closed is True


def test_periodic_dev_eval_runner_records_heuristic_policy_alignment(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    env = FakeEvalEnv(
        reset_batch=make_decision_batch(train_script.DecisionBoundaryBatch, terminal=False),
        step_batch=make_decision_batch(train_script.DecisionBoundaryBatch, terminal=True, decision_id=1),
    )
    model = RecordingEvalModel()
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    class FirstLegalHeuristic:
        def choose_action(self, _obs, legal_ids):
            return int(legal_ids[0])

    runner = train_script._PeriodicDevEvalRunner(
        stack=SimpleNamespace(),
        model=model,
        opponent_policy_id="b2_heuristicpublic",
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="focal",
        heuristic_policy=FirstLegalHeuristic(),
        require_sorted_legal_ids=False,
    )
    scheduled_game = make_scheduled_game(
        train_script.ScheduledGame,
        focal_policy_id="focal",
        opponent_policy_id="b2_heuristicpublic",
    )

    runner.run_game(scheduled_game)
    summary = runner.policy_alignment_summary()

    assert summary is not None
    assert summary["schema"] == "policy_alignment_diagnostics_v1"
    assert summary["model_policy_id"] == "focal"
    assert summary["reference_policy_id"] == "b2_heuristicpublic"
    assert summary["all_decisions"]["compared_steps"] == 1
