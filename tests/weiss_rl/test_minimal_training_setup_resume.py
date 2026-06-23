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


def test_build_minimal_training_setup_restores_offset_and_builds_runtime(tmp_path: Path) -> None:
    scalars_path = tmp_path / "training_metrics.jsonl"
    scalars_path.write_text(
        '{"update_count": 3, "init_schedule_offset_updates": 17}\n',
        encoding="utf-8",
    )
    training_paths = minimal_training_paths(tmp_path)
    rewards_config = object()
    stack = minimal_training_stack(
        tmp_path=tmp_path,
        algorithm=" ppo ",
        recurrent_core="lstm",
        encoder_kind="typed",
        rewards_config=rewards_config,
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    runtime_mode = object()
    b1_baseline_run_dir = tmp_path / "b1"
    seed_snapshot_run_dir = tmp_path / "seed_source"
    device = object()
    model = FakeModel()
    learner = FakeLearner(model)
    compiled_model = object()
    runtime_config = object()
    resume_state = SimpleNamespace(
        checkpoint_path=tmp_path / "resume.pt",
        update_count=5,
        policy_version=7,
        init_schedule_offset_updates=0,
    )
    recorder = MinimalSetupHookRecorder(
        training_paths=training_paths,
        model=model,
        learner=learner,
        compiled_model=compiled_model,
        runtime_config=runtime_config,
        resume_state=resume_state,
        config_hash256="config-hash",
        canonical_config={"config": "canonical"},
        fail_initialize=True,
    )

    setup = build_minimal_training_setup(
        stack=stack,
        contract=minimal_training_contract(include_observation=True),
        artifacts=artifacts,
        num_envs=4,
        unroll_length=8,
        profile="fast",
        device=device,
        seed=99,
        checkpoint_interval_updates=5,
        spec_hash256="spec-hash",
        runtime_mode=runtime_mode,
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
        hooks=recorder.hooks(),
    )

    assert setup.observation_dim == 3
    assert setup.action_dim == 9
    assert setup.training_config is stack.config.training
    assert setup.rewards_config is rewards_config
    assert setup.training_paths is training_paths
    assert setup.pass_action_id == 8
    assert setup.algorithm == "ppo"
    assert setup.model is model
    assert setup.learner is learner
    assert setup.latest_metrics == {"snapshot_publish_latency_ms": 1.0}
    assert setup.init_schedule_offset_updates == 17
    assert setup.resume_state is resume_state
    assert setup.config_hash256 == "config-hash"
    assert learner.init_schedule_offset_updates == 17
    assert [call[0] for call in recorder.calls] == [
        "spec_dimensions",
        "training_paths",
        "validate",
        "build_model",
        "model_to",
        "compile",
        "build_learner",
        "restore",
        "config_hash",
        "baseline",
        "state_dict",
        "canonical",
        "seed_snapshot",
        "runtime_config",
        "runtime",
        "snapshot",
    ]
    assert recorder.calls[2][1] == {"algorithm": "ppo", "recurrent_core": "lstm", "encoder_kind": "typed"}
    assert recorder.calls[3][1]["observation_dim"] == 3
    assert recorder.calls[3][1]["observation_spec"] == {"shape": [3]}
    assert recorder.calls[6][1]["compiled_model"] is compiled_model
    assert recorder.calls[6][1]["pass_action_id"] == 8
    assert recorder.calls[7][1]["expected_spec_hash256"] == "spec-hash"
    assert recorder.calls[9][1]["baseline_run_dir"] == b1_baseline_run_dir
    assert recorder.calls[12][1]["expected_model_state_dict"] == {"weight": 1}
    assert recorder.calls[12][1]["expected_config_canonical"] == {"config": "canonical"}
    assert recorder.calls[13][1] == {
        "stack": stack,
        "num_envs": 4,
        "unroll_length": 8,
        "profile": "fast",
        "seed": 99,
        "pass_action_id": 8,
        "runtime_mode": runtime_mode,
    }
    assert recorder.calls[14][1]["config"] is runtime_config
    assert recorder.calls[14][1]["performance_log_path"] == training_paths.performance_log_path
    assert recorder.calls[15][1] == {"learner_model": model, "learner_update_count": 5, "force": True}
