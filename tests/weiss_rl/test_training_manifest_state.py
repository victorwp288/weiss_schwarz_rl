from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.train_entrypoint.cli import TrainCliState, TrainStartupState, prepare_train_manifest_state


def test_train_manifest_state_writes_reports_and_logs_tensorboard_context(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    writes: list[tuple[Path, dict[str, object]]] = []
    training_config = object()
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"report_eval": tmp_path / "seeds.txt"},
        config=SimpleNamespace(
            training=training_config,
            reproducibility=None,
            system=SimpleNamespace(actor_device="cuda:1"),
        ),
    )
    cli = TrainCliState(
        run_label="manifest_run",
        num_envs=4,
        unroll_length=8,
        max_updates=3,
        runtime_mode="train_ordered",
        stack=stack,
        training_config=training_config,
        manifest_only_reason=None,
        public_demo_enabled=False,
        resume_run_dir=None,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=tmp_path / "init.pt",
        init_schedule_offset_override_updates=12,
    )
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=object(),
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "compat"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="manifest_run",
        resume_artifacts=None,
    )
    layout = SimpleNamespace(tensorboard_dir=tmp_path / "tb")
    artifacts = SimpleNamespace(
        run_dir=tmp_path / "runs" / "manifest_run",
        run_dir_name="manifest_run",
        layout=layout,
        manifest_path=tmp_path / "runs" / "manifest_run" / "manifest.json",
        run_summary_path=tmp_path / "runs" / "manifest_run" / "run_summary.json",
        determinism_report_path=tmp_path / "runs" / "manifest_run" / "determinism.json",
        environment_path=tmp_path / "runs" / "manifest_run" / "environment.json",
    )

    class FakeManifest:
        hardware = {"device": "cuda:0"}

        def __init__(self, **kwargs: object) -> None:
            calls["manifest_kwargs"] = kwargs

        def to_dict(self) -> dict[str, object]:
            return {"manifest": "payload"}

    class FakeTensorBoardLogger:
        enabled = True

        def __init__(self, log_dir: Path) -> None:
            calls["tensorboard_dir"] = log_dir

        def log_run_context(self, **kwargs: object) -> None:
            calls["tensorboard_context"] = kwargs

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        writes.append((path, dict(payload)))

    def fake_write_run_artifacts(runs_dir: Path, manifest: object, *, run_label: str | None) -> object:
        calls["write_run_artifacts"] = (runs_dir, manifest, run_label)
        return artifacts

    api = SimpleNamespace(
        _resolve_device=lambda loaded_stack, _device_arg: "cuda:0",
        _resolve_runtime_profile=lambda loaded_stack, _profile_arg: "fast",
        _resolve_seed=lambda loaded_stack, _seed_arg: 99,
        _manifest_actor_device_layout=lambda **kwargs: calls.setdefault("actor_layout", kwargs) or ("cuda:1",),
        _resolve_policy_set_selection=lambda loaded_stack, **kwargs: (
            ["B0 RandomLegal"],
            {"status": "resolved", "mode": "deterministic_v1"},
        ),
        RunManifest=FakeManifest,
        _git_dirty=lambda: False,
        canonical_config_dict=lambda loaded_stack: {"config": "canonical"},
        build_seed_file_manifest=lambda seed_sets, *, root: {"report_eval": {"path": "seeds.txt"}},
        _hardware_summary=lambda device, *, actor_device, actor_device_layout: {
            "device": device,
            "actor_device": actor_device,
            "actor_device_layout": actor_device_layout,
        },
        _evaluation_pinning=lambda loaded_stack: {"eval_device": "cpu"},
        write_run_artifacts=fake_write_run_artifacts,
        _load_json_object=lambda path, *, label: {"label": label},
        augment_run_summary_payload=lambda payload, **kwargs: payload.update({"run_summary": kwargs}),
        augment_determinism_payload=lambda payload, **kwargs: payload.update({"determinism": kwargs}),
        augment_environment_payload=lambda payload, **kwargs: payload.update({"environment": kwargs}),
        _write_json=fake_write_json,
        TensorBoardLogger=FakeTensorBoardLogger,
        tensorboard_unavailable_reason=lambda: None,
        sys=SimpleNamespace(argv=["python", "train.py"], stderr=SimpleNamespace(write=lambda text: None)),
    )
    args = SimpleNamespace(
        device="cuda",
        profile="fast",
        seed=99,
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        seed_snapshot_run_dir=tmp_path / "seed_source",
    )

    manifest_state = prepare_train_manifest_state(args=args, api=api, startup=startup)

    assert manifest_state.artifacts is artifacts
    assert manifest_state.device == "cuda:0"
    assert manifest_state.profile == "fast"
    assert manifest_state.seed == 99
    assert manifest_state.policy_set_selection_details == {"status": "resolved", "mode": "deterministic_v1"}
    assert calls["write_run_artifacts"][0] == tmp_path / "runs"
    assert calls["write_run_artifacts"][2] == "manifest_run"
    assert calls["manifest_kwargs"]["run_id256"] == "a" * 64
    assert calls["manifest_kwargs"]["seed_derivation"]["effective_base_seed64"] == 99
    assert calls["manifest_kwargs"]["seed_derivation"]["cli_seed_override"] is True
    assert calls["manifest_kwargs"]["hardware"]["actor_device"] == "cuda:1"
    assert calls["tensorboard_dir"] == tmp_path / "tb"
    assert calls["tensorboard_context"]["manifest"] == {"manifest": "payload"}
    assert writes == [
        (artifacts.run_summary_path, manifest_state.run_summary_payload),
        (artifacts.determinism_report_path, manifest_state.determinism_payload),
        (artifacts.environment_path, manifest_state.environment_payload),
    ]
    assert manifest_state.run_summary_payload["run_summary"]["init_from_checkpoint_path"] == tmp_path / "init.pt"
    assert manifest_state.determinism_payload["determinism"]["resume_checkpoint_path"] == tmp_path / "resume.pt"
    assert manifest_state.environment_payload["environment"]["argv"] == ["python", "train.py"]
