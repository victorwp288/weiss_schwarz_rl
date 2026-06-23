from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.train_entrypoint.cli import TrainCliState, TrainStartupState, execute_train_run


def test_train_execution_dispatch_stages_public_demo_without_training(tmp_path: Path, capsys) -> None:
    calls: dict[str, object] = {}
    cli = TrainCliState(
        run_label="toy_demo",
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_ordered",
        stack=SimpleNamespace(root=tmp_path),
        training_config=object(),
        manifest_only_reason=None,
        public_demo_enabled=True,
        resume_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
    )
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=None,
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "public_demo"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="toy_demo",
        resume_artifacts=None,
    )
    manifest_state = SimpleNamespace(
        artifacts=SimpleNamespace(run_dir=tmp_path / "runs" / "toy_demo"),
        device="cpu",
        profile="default",
        seed=7,
        tensorboard_logger=SimpleNamespace(),
    )

    def fake_stage_public_demo_run(run_dir: Path) -> object:
        calls["stage"] = run_dir
        return SimpleNamespace(policy_ids=["B0 RandomLegal"], catalog_path=run_dir / "public_demo" / "catalog.json")

    api = SimpleNamespace(
        PUBLIC_DEMO_MODE="public_demo",
        stage_public_demo_run=fake_stage_public_demo_run,
        _run_minimal_training=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("public demo must not execute simulator training")
        ),
    )

    execute_train_run(args=SimpleNamespace(), api=api, startup=startup, manifest_state=manifest_state)

    assert calls["stage"] == tmp_path / "runs" / "toy_demo"
    stdout = capsys.readouterr().out
    assert "Staged public-demo toy catalog and policy bundle" in stdout
    assert "demo-only" in stdout


def test_train_execution_dispatch_runs_minimal_training_with_resolved_settings(tmp_path: Path, capsys) -> None:
    calls: dict[str, object] = {}
    training_config = object()
    tensorboard_logger = SimpleNamespace()
    cli = TrainCliState(
        run_label="train_run",
        num_envs=4,
        unroll_length=8,
        max_updates=3,
        runtime_mode="train_async_fast",
        stack=SimpleNamespace(root=tmp_path),
        training_config=training_config,
        manifest_only_reason=None,
        public_demo_enabled=False,
        resume_run_dir=None,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=tmp_path / "init_cli.pt",
        init_schedule_offset_override_updates=12,
    )
    contract = object()
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=contract,
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "runtime"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="train_run",
        resume_artifacts=None,
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "runs" / "train_run")
    manifest_state = SimpleNamespace(
        artifacts=artifacts,
        device="cuda:0",
        profile="fast",
        seed=99,
        tensorboard_logger=tensorboard_logger,
    )
    execution_settings = SimpleNamespace(
        checkpoint_interval_updates=5,
        b1_baseline_run_dir=tmp_path / "b1",
        seed_snapshot_run_dir=tmp_path / "seed_source",
        init_from_checkpoint_path=tmp_path / "init_resolved.pt",
        profile_timers=True,
        torch_profiler=False,
    )

    def fake_minimal_training(**kwargs: object) -> dict[str, float]:
        calls["minimal"] = kwargs
        return {"loss": 1.25, "policy_loss": 0.5, "value_loss": 0.75, "entropy": 0.125}

    def fake_resolve_training_execution_settings(**kwargs: object) -> object:
        calls["settings"] = kwargs
        return execution_settings

    api = SimpleNamespace(
        _runtime_training_prerequisite_failure=lambda stack: None,
        _raise_runtime_prerequisite_failure=lambda reason: (_ for _ in ()).throw(RuntimeError(reason)),
        _noleague_training_prerequisite_failure=lambda stack: None,
        _raise_noleague_training_prerequisite_failure=lambda reason: (_ for _ in ()).throw(RuntimeError(reason)),
        resolve_training_execution_settings=fake_resolve_training_execution_settings,
        profiling_enabled_message=lambda config: "Profiling enabled",
        _run_minimal_training=fake_minimal_training,
    )
    args = SimpleNamespace(
        checkpoint_interval_updates=5,
        b1_baseline_run_dir=tmp_path / "b1_cli",
        seed_snapshot_run_dir=tmp_path / "seed_cli",
        init_from_checkpoint=tmp_path / "init_cli.pt",
    )

    execute_train_run(args=args, api=api, startup=startup, manifest_state=manifest_state)

    assert calls["settings"] == {
        "training_config": training_config,
        "checkpoint_interval_override": 5,
        "b1_baseline_run_dir": tmp_path / "b1_cli",
        "seed_snapshot_run_dir": tmp_path / "seed_cli",
        "init_from_checkpoint": tmp_path / "init_cli.pt",
    }
    minimal = calls["minimal"]
    assert minimal["stack"] is cli.stack
    assert minimal["contract"] is contract
    assert minimal["artifacts"] is artifacts
    assert minimal["num_envs"] == 4
    assert minimal["unroll_length"] == 8
    assert minimal["max_updates"] == 3
    assert minimal["profile"] == "fast"
    assert minimal["device"] == "cuda:0"
    assert minimal["seed"] == 99
    assert minimal["checkpoint_interval_updates"] == 5
    assert minimal["run_id256"] == "a" * 64
    assert minimal["config_hash256"] == "c" * 64
    assert minimal["spec_hash256"] == "b" * 64
    assert minimal["runtime_mode"] == "train_async_fast"
    assert minimal["b1_baseline_run_dir"] == tmp_path / "b1"
    assert minimal["seed_snapshot_run_dir"] == tmp_path / "seed_source"
    assert minimal["profile_timers"] is True
    assert minimal["torch_profiler"] is False
    assert minimal["resume_checkpoint_path"] == tmp_path / "resume.pt"
    assert minimal["init_from_checkpoint_path"] == tmp_path / "init_resolved.pt"
    assert minimal["init_schedule_offset_override_updates"] == 12
    assert minimal["tensorboard_logger"] is tensorboard_logger
    stdout = capsys.readouterr().out
    assert "Profiling enabled" in stdout
    assert "Completed canonical single-node training run: loss=1.250000" in stdout
