from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.loop.setup import build_minimal_training_setup

from .minimal_training_setup_test_support import (
    FakeLearner,
    FakeModel,
    MinimalSetupHookRecorder,
    minimal_training_contract,
    minimal_training_paths,
    minimal_training_stack,
)


def test_build_minimal_training_setup_init_checkpoint_override_sets_schedule_offset(tmp_path: Path) -> None:
    training_paths = minimal_training_paths(tmp_path)
    model = FakeModel()
    learner = FakeLearner(model)
    init_state = SimpleNamespace(
        checkpoint_path=tmp_path / "init.pt",
        update_count=20,
        init_schedule_offset_updates=30,
        policy_version=4,
    )
    recorder = MinimalSetupHookRecorder(
        training_paths=training_paths,
        model=model,
        learner=learner,
        init_state=init_state,
        config_hash256="config",
        fail_restore=True,
        fail_seed_snapshot=True,
    )

    setup = build_minimal_training_setup(
        stack=minimal_training_stack(tmp_path=tmp_path),
        contract=minimal_training_contract(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        num_envs=1,
        unroll_length=1,
        profile="default",
        device=object(),
        seed=1,
        checkpoint_interval_updates=2,
        spec_hash256="spec",
        runtime_mode=object(),
        b1_baseline_run_dir=None,
        seed_snapshot_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=tmp_path / "init.pt",
        init_schedule_offset_override_updates=12,
        hooks=recorder.hooks(),
    )

    assert setup.init_schedule_offset_updates == 12
    assert learner.init_schedule_offset_updates == 12
    assert "restore" not in [call[0] for call in recorder.calls]
    init_call = next(call for call in recorder.calls if call[0] == "init")
    assert init_call[1]["checkpoint_path"] == tmp_path / "init.pt"
