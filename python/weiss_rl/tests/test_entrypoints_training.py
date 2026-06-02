from __future__ import annotations

from .test_entrypoints import (
    Path,
    _copy_repo_configs,
    _patch_periodic_dev_eval_config,
    _run_entrypoint,
    _run_public_demo_train,
    _write_b1_baseline_run_fixture,
    _write_eval_only_stack_config,
    _write_manifest_only_stack_config,
    _write_policy_set_inputs,
    _write_runtime_weiss_sim,
    _write_stub_weiss_sim,
    json,
    public_demo_spec_hash256,
    spec_bundle_hash,
    torch,
)


def test_train_entrypoint_fails_fast_on_runtime_spec_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="999",
        run_label="mismatch_run",
    )

    assert result.returncode != 0
    assert "Spec mismatch" in result.stderr


def test_train_entrypoint_rejects_invalid_runtime_spec_bundle_before_claiming_verification(tmp_path: Path) -> None:
    invalid_bundle = {
        "policy_version": 3,
        "spec_hash": 123,
        "observation": {"obs_encoding_version": 2, "dtype": "i32", "obs_len": 512},
        "action": {"action_encoding_version": 1, "pass_action_id": 8},
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                f"    return {invalid_bundle!r}",
                "",
            )
        ),
        encoding="utf-8",
    )
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="123",
        run_label="invalid_spec_bundle",
    )

    assert result.returncode != 0
    assert "invalid spec_bundle payload" in result.stderr
    assert "Verified runtime spec bundle" not in result.stdout


def test_train_entrypoint_persists_runtime_spec_bundle(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="spec_bundle_run",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "spec_bundle_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["simulator"]["compatibility_hash"] == "123"
    assert manifest["spec_bundle"] == bundle
    assert manifest["policy_set_selection"] == []
    assert manifest["policy_set_selection_details"] == {
        "mode": "not_configured",
        "status": "not_configured",
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
    }
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)
    assert "computed_run_id64:" in result.stdout
    assert "computed_run_id256:" in result.stdout
    assert "run_label:              spec_bundle_run" in result.stdout
    assert "run_dir_name:           spec_bundle_run" in result.stdout
    assert "Manifest scaffold only: no learner training or rollout collection was executed." in result.stdout
    assert "missing config blocks: environment, training, model" in result.stdout


def test_train_entrypoint_locked_stack_fails_on_incomplete_runtime(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="locked_stack_requires_runtime",
    )

    assert result.returncode != 0
    assert "Canonical simulator-backed training requires a weiss_sim runtime with stepping support" in result.stderr
    assert "active weiss_sim runtime is missing stepping APIs" in result.stderr


def test_train_entrypoint_resolves_policy_set_selection_when_inputs_are_supplied(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="resolved_policy_set_run",
        extra_args=[
            "--snapshot-registry-json",
            str(snapshot_registry_path),
            "--dev-eval-summaries-json",
            str(dev_eval_summaries_path),
        ],
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "resolved_policy_set_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["policy_set_selection"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    assert manifest["policy_set_selection_details"] == {
        "mode": "deterministic_v1",
        "status": "resolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": "policy_set_snapshot_registry.json",
            "dev_eval_summaries_json": "policy_set_dev_eval_summaries.json",
        },
        "missing_inputs": [],
        "selected_policy_count": 10,
    }


def test_train_entrypoint_uses_default_run_dir_when_no_label_override(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="123",
    )

    assert result.returncode == 0, result.stderr
    manifest_path_line = next(line for line in result.stdout.splitlines() if line.startswith("Wrote manifest: "))
    manifest_path = Path(manifest_path_line.removeprefix("Wrote manifest: ").strip())
    assert manifest_path.name == "manifest.json"
    assert manifest_path.parent.name.startswith("run_")
    assert "run_label:              (default)" in result.stdout
    assert f"run_dir_name:           {manifest_path.parent.name}" in result.stdout


def test_train_entrypoint_accepts_deprecated_run_id_alias(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="123",
        run_id_alias="compat_alias_run",
    )

    assert result.returncode == 0, result.stderr
    assert "deprecated; use --run-label instead" in result.stderr
    assert (tmp_path / "runs" / "compat_alias_run" / "manifest.json").is_file()


def test_train_entrypoint_runs_periodic_dev_eval_and_handles_empty_ids_pass_fallback(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(
        tmp_path,
        spec_hash=123,
        pass_action_id=3,
        empty_eval_legal_row=True,
    )
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_run",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "Periodic dev eval: update=1 opponent=b0_randomlegal" in result.stdout

    eval_root = tmp_path / "runs" / "periodic_dev_eval_run" / "eval" / "dev_eval" / "update_1"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    diagnostics_payload = json.loads((eval_root / "b0_randomlegal" / "diagnostics.json").read_text(encoding="utf-8"))
    episodes_lines = (eval_root / "b0_randomlegal" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()

    assert seed_usage["seed_file"]["path"] == "configs/seeds/dev_eval_seeds.txt"
    assert seed_usage["paired_seed_count"] == 1
    assert seed_usage["paired_seeds"] == [7]
    assert seed_usage["focal_policy"]["update_count"] == 1
    assert seed_usage["focal_policy"]["policy_version"] == 1
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert len(episodes_lines) == 2
    assert summary_payload["summary"]["games"] == 2
    assert summary_payload["evaluation_context"] == {
        "artifact_scope": "periodic_dev_eval",
        "update_count": 1,
        "policy_version": 1,
        "checkpoint_path": "training/checkpoints/checkpoint_1.pt",
        "seed_usage_path": "eval/dev_eval/update_1/b0_randomlegal/seed_usage.json",
        "anchor_display_name": "B0 RandomLegal",
    }
    assert diagnostics_payload["seat_results"]["seat0_wins"] == 2
    assert diagnostics_payload["seat_results"]["seat1_wins"] == 0
    assert (eval_root / "b0_randomlegal" / "matchup_summary.csv").is_file()


def test_train_entrypoint_periodic_dev_eval_writes_exact_current_checkpoint(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_checkpoint_traceability",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "2",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr

    run_root = tmp_path / "runs" / "periodic_dev_eval_checkpoint_traceability"
    eval_root = run_root / "eval" / "dev_eval" / "update_1"
    checkpoint_path = run_root / "training" / "checkpoints" / "checkpoint_1.pt"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint_path.is_file()
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert summary_payload["evaluation_context"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert checkpoint_payload["update_count"] == 1


def test_train_entrypoint_uses_configured_checkpoint_interval_by_default(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="checkpoint_default_from_config",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / "checkpoint_default_from_config"
    registry = json.loads((run_root / "training" / "snapshots" / "registry.json").read_text(encoding="utf-8"))
    assert [snapshot["policy_id"] for snapshot in registry["snapshots"]] == ["b1_noleague_baseline"]
    assert not (run_root / "eval" / "promotion_gate" / "update_1").exists()


def test_train_entrypoint_public_demo_accepts_profile_timers_flag(tmp_path: Path) -> None:
    stack_config = _copy_repo_configs(tmp_path)
    run_label = "toy_public_demo_profile_timers"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        run_label=run_label,
        extra_args=["--public-demo", "--profile-timers"],
    )

    assert result.returncode == 0, result.stderr
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    run_summary = json.loads((tmp_path / "runs" / run_label / "run_summary.json").read_text(encoding="utf-8"))
    training_controls = run_summary["training_controls"]
    assert training_controls["profile_timers"] is True
    assert training_controls["torch_profiler"] is False


def test_train_entrypoint_profile_timers_does_not_emit_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "profile_timers_no_trace"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--profile-timers",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert not (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is True
    assert run_summary["training_controls"]["torch_profiler"] is False
    assert determinism["training_controls"]["profile_timers"] is True
    assert determinism["training_controls"]["torch_profiler"] is False


def test_train_entrypoint_emits_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "torch_profiler_trace"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--torch-profiler",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is False
    assert run_summary["training_controls"]["torch_profiler"] is True
    assert determinism["training_controls"]["profile_timers"] is False
    assert determinism["training_controls"]["torch_profiler"] is True


def test_train_entrypoint_public_demo_stages_public_safe_catalog_without_weiss_sim(tmp_path: Path) -> None:
    result, run_dir = _run_public_demo_train(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((run_dir / "public_demo" / "catalog.json").read_text(encoding="utf-8"))
    policies = json.loads((run_dir / "public_demo" / "policy_manifest.json").read_text(encoding="utf-8"))
    scalars_lines = (run_dir / "training" / "logs" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["simulator"]["runtime"] == "public_demo"
    assert manifest["simulator"]["public_safe"] is True
    assert manifest["spec_bundle"]["action"]["action_space_size"] == 9
    assert catalog["public_safe"] is True
    assert len(catalog["card_pool"]) == 12
    assert len(catalog["decks"]) == 3
    assert policies["policy_ids"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "toy_policy_000100",
        "toy_policy_000200",
    ]
    assert len(scalars_lines) == 1
    assert "Loaded synthetic public-demo spec bundle" in result.stdout
    assert "Verified runtime spec bundle" not in result.stdout
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    assert "demo-only" in result.stdout
