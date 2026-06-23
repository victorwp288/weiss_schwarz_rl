from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.learners.impala import ImpalaLearner

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _load_train_script_module,
    _make_policy_value_model,
)


def test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    artifacts = train_script._run_artifacts_from_existing_run_dir(run_dir)
    alias_stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=stack.config.curriculum,
            evaluation=SimpleNamespace(periodic_dev_eval_interval_updates=25),
        )
    )

    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 3
    learner.policy_version = 2
    learner.total_samples_processed = 96
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_3.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    tracker = train_script._publish_checkpoint_aliases(
        stack=alias_stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.25},
    )
    assert training_paths.latest_checkpoint_path.is_file()
    assert not training_paths.best_checkpoint_path.is_file()
    assert tracker["latest"]["metric_kind"] is None
    assert tracker["best"] is None

    learner.update_count = 4
    learner.policy_version = 3
    learner.total_samples_processed = 128
    second_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_4.pt"
    train_script._write_checkpoint(
        checkpoint_path=second_checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    tracker = train_script._publish_checkpoint_aliases(
        stack=alias_stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=second_checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.5},
        dev_eval_summary={"aggregate_score": 0.61, "uncertainty": {"mean": 0.61}},
    )
    assert tracker["best"]["metric_kind"] == "dev_eval_mean"
    assert tracker["best"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_4.pt")

    restored_learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.best_checkpoint_path,
        learner=restored_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    assert resume_state.update_count == 4
    assert resume_state.policy_version == 3
    assert resume_state.total_samples_processed == 128
    assert restored_learner.update_count == 4
    assert restored_learner.policy_version == 3

    restored_learner.update_count = 99
    restored_learner.policy_version = 77
    restored_learner.total_samples_processed = 12345
    preserved_start_time = restored_learner.start_time
    preserved_resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.best_checkpoint_path,
        learner=restored_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        restore_counters=False,
    )
    assert preserved_resume_state.update_count == 99
    assert preserved_resume_state.policy_version == 77
    assert preserved_resume_state.total_samples_processed == 12345
    assert restored_learner.update_count == 99
    assert restored_learner.policy_version == 77
    assert restored_learner.start_time == preserved_start_time
