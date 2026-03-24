from __future__ import annotations

from pathlib import Path

from weiss_rl.learners.impala_learner import ImpalaLearner


def test_impala_learner_writes_checkpoints_using_update_count(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
    )

    for _ in range(4):
        result = learner.update({})
        assert result["loss"] == 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_2.pt").is_file()
    assert (checkpoint_dir / "checkpoint_4.pt").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_checkpoint_records_update_and_policy_version(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=3,
    )

    for _ in range(3):
        learner.update({})

    checkpoint_text = (tmp_path / "checkpoints" / "checkpoint_3.pt").read_text(encoding="utf-8")
    assert checkpoint_text == "update_count: 3\npolicy_version: 1\n"
