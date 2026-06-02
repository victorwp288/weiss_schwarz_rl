from __future__ import annotations

from .test_entrypoints import (
    ArtifactLayout,
    Path,
    _copy_repo_configs,
    _write_eval_only_stack_config,
    _write_policy_set_inputs,
    json,
    load_stack_config,
    pytest,
    shutil,
)


def test_eval_entrypoint_prefers_run_local_policy_selection_over_manifest_fallback(tmp_path: Path) -> None:
    import weiss_rl.workflows.eval_entrypoint as eval_entrypoint
    from weiss_rl.workflows import (
        eval_canonical,
        eval_canonical_cli_messages,
        eval_canonical_dependencies,
        eval_canonical_entrypoint_adapter,
        eval_canonical_entrypoint_request,
        eval_canonical_figure_outputs,
        eval_canonical_final_eval,
        eval_canonical_metagame_outputs,
        eval_canonical_output_bundle,
        eval_canonical_outputs,
        eval_canonical_phases,
        eval_canonical_policy_runtime,
        eval_canonical_publisher,
        eval_canonical_readiness_outputs,
        eval_canonical_report_publication,
        eval_canonical_runtime,
        eval_canonical_seed_budget,
        eval_canonical_setup,
        eval_canonical_state,
        eval_canonical_supplemental_outputs,
        eval_canonical_tensorboard_publication,
        eval_dispatch,
        eval_dispatch_dependencies,
        eval_dispatch_request,
        eval_dispatch_route_adapters,
        eval_dispatch_routes,
        eval_entrypoint_compat,
        eval_entrypoint_export_groups,
        eval_entrypoint_exports,
        eval_entrypoint_external_exports,
        eval_entrypoint_main,
        eval_entrypoint_report_exports,
        eval_entrypoint_runtime,
        eval_entrypoint_workflow_exports,
        eval_modes,
        eval_parser,
        eval_public_demo_mode,
        eval_reports,
        eval_startup,
        eval_startup_dependencies,
        eval_startup_prepare,
        eval_startup_state,
        eval_startup_validation,
        eval_summary_mode,
    )
    from weiss_rl.workflows import eval_entrypoint as eval_script

    assert eval_script is eval_entrypoint
    assert (
        eval_entrypoint.run_entrypoint_canonical_eval_pipeline
        is eval_entrypoint_compat.run_entrypoint_canonical_eval_pipeline
    )
    assert eval_entrypoint.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_REPORT_HELPER_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_WORKFLOW_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ADDITIONAL_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS == [
        *eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS,
        *eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS,
    ]
    assert set(eval_entrypoint_exports.__all__) == set(
        [
            *eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS,
            *eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS,
            "EVAL_ADDITIONAL_COMPAT_EXPORTS",
            "EVAL_REPORT_HELPER_EXPORTS",
            "EVAL_WORKFLOW_COMPAT_EXPORTS",
        ]
    )
    assert eval_entrypoint.run_eval_entrypoint_main is eval_entrypoint_main.run_eval_entrypoint_main
    assert eval_entrypoint.run_eval_entrypoint is eval_entrypoint_runtime.run_eval_entrypoint
    assert (
        eval_entrypoint.run_eval_entrypoint_canonical_pipeline
        is eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline
    )
    assert eval_script.build_eval_parser is eval_parser.build_eval_parser
    assert eval_entrypoint_exports.build_eval_parser is eval_entrypoint_workflow_exports.build_eval_parser
    assert eval_entrypoint_exports.resolve_eval_policies is eval_entrypoint_external_exports.resolve_eval_policies
    assert eval_entrypoint_exports._resolve_policy_ids_for_run is (
        eval_entrypoint_report_exports._resolve_policy_ids_for_run
    )
    assert eval_entrypoint_exports.TensorBoardLogger is eval_entrypoint_external_exports.TensorBoardLogger
    assert eval_entrypoint_exports.CanonicalEvalDependencies is (
        eval_entrypoint_workflow_exports.CanonicalEvalDependencies
    )
    assert eval_script.run_canonical_eval_pipeline is eval_canonical.run_canonical_eval_pipeline
    assert (
        eval_script.run_canonical_eval_entrypoint_pipeline
        is eval_canonical_entrypoint_adapter.run_canonical_eval_entrypoint_pipeline
    )
    assert "CanonicalEvalEntrypointRequest" in eval_canonical_entrypoint_request.__all__
    assert "canonical_eval_entrypoint_request" in eval_canonical_entrypoint_request.__all__
    assert eval_script.CanonicalEvalDependencies is eval_canonical.CanonicalEvalDependencies
    assert eval_script.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical_phases.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical_phases.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical.prepare_canonical_eval_run_state is eval_canonical_setup.prepare_canonical_eval_run_state
    assert eval_canonical_phases.prepare_canonical_eval_run_state is (
        eval_canonical_setup.prepare_canonical_eval_run_state
    )
    assert eval_canonical.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical_phases.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical.write_canonical_eval_outputs is eval_canonical_outputs.write_canonical_eval_outputs
    assert eval_canonical_phases.write_canonical_eval_outputs is (eval_canonical_outputs.write_canonical_eval_outputs)
    assert "render_canonical_eval_output_messages" in eval_canonical_cli_messages.__all__
    assert "build_canonical_figure_outputs" in eval_canonical_figure_outputs.__all__
    assert "run_canonical_final_eval_output" in eval_canonical_final_eval.__all__
    assert "build_canonical_metagame_output" in eval_canonical_metagame_outputs.__all__
    assert "build_canonical_readiness_output" in eval_canonical_readiness_outputs.__all__
    assert "build_canonical_supplemental_outputs" in eval_canonical_supplemental_outputs.__all__
    assert "build_canonical_eval_output_bundle" in eval_canonical_output_bundle.__all__
    assert "publish_canonical_eval_outputs" in eval_canonical_publisher.__all__
    assert "publish_canonical_eval_run_reports" in eval_canonical_report_publication.__all__
    assert "publish_canonical_eval_tensorboard_summaries" in eval_canonical_tensorboard_publication.__all__
    assert "resolve_canonical_eval_policy_runtime" in eval_canonical_policy_runtime.__all__
    assert "resolve_recommended_focal_policy_id" in eval_canonical_policy_runtime.__all__
    assert "resolve_canonical_eval_seed_budget" in eval_canonical_seed_budget.__all__
    assert eval_script.run_eval_dispatch is eval_dispatch_routes.run_eval_dispatch
    assert eval_dispatch.run_eval_dispatch is eval_dispatch_routes.run_eval_dispatch
    assert eval_script.EvalDispatchDependencies is eval_dispatch_dependencies.EvalDispatchDependencies
    assert eval_dispatch.EvalDispatchDependencies is eval_dispatch_dependencies.EvalDispatchDependencies
    assert eval_dispatch.EvalDispatchRequest is eval_dispatch_request.EvalDispatchRequest
    assert "EvalDispatchRequest" in eval_dispatch_request.__all__
    assert "run_canonical_eval_request" in eval_dispatch_route_adapters.__all__
    assert eval_script.build_eval_dispatch_dependencies is eval_dispatch_dependencies.build_eval_dispatch_dependencies
    assert eval_dispatch.build_eval_dispatch_dependencies is eval_dispatch_dependencies.build_eval_dispatch_dependencies
    assert eval_dispatch._print_startup_verification is eval_dispatch_route_adapters._print_startup_verification
    assert eval_dispatch.run_public_demo_eval_route is eval_dispatch_route_adapters.run_public_demo_eval_route
    assert eval_dispatch.run_canonical_eval_route is eval_dispatch_route_adapters.run_canonical_eval_route
    assert eval_dispatch.run_summary_only_eval_route is eval_dispatch_route_adapters.run_summary_only_eval_route
    assert eval_script.EvalStartup is eval_startup_state.EvalStartup
    assert eval_script.EvalStartupDependencies is eval_startup_dependencies.EvalStartupDependencies
    assert eval_startup.EvalStartupDependencies is eval_startup_dependencies.EvalStartupDependencies
    assert eval_script.build_eval_startup_dependencies is eval_startup_dependencies.build_eval_startup_dependencies
    assert eval_startup.build_eval_startup_dependencies is eval_startup_dependencies.build_eval_startup_dependencies
    assert eval_script.EvalValidatedArgs is eval_startup_state.EvalValidatedArgs
    assert eval_startup.EvalStartup is eval_startup_state.EvalStartup
    assert eval_startup.EvalValidatedArgs is eval_startup_state.EvalValidatedArgs
    assert eval_script.prepare_eval_startup is eval_startup_prepare.prepare_eval_startup
    assert eval_startup.prepare_eval_startup is eval_startup_prepare.prepare_eval_startup
    assert eval_script.validate_eval_args is eval_startup_validation.validate_eval_args
    assert eval_startup.validate_eval_args is eval_startup_validation.validate_eval_args
    assert eval_script.run_public_demo_eval_mode is eval_public_demo_mode.run_public_demo_eval_mode
    assert eval_modes.run_public_demo_eval_mode is eval_public_demo_mode.run_public_demo_eval_mode
    assert eval_script.run_summary_only_eval_mode is eval_summary_mode.run_summary_only_eval_mode
    assert eval_modes.run_summary_only_eval_mode is eval_summary_mode.run_summary_only_eval_mode
    assert eval_script._resolve_policy_ids_for_run is eval_reports._resolve_policy_ids_for_run
    assert eval_script._load_run_summary_or_default is eval_reports._load_run_summary_or_default

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
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
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_report_facade_reexports_split_module_owners() -> None:
    from weiss_rl.workflows import (
        eval_policy_final_set_resolution,
        eval_policy_manifest_selection,
        eval_policy_selection,
        eval_policy_selection_results,
        eval_report_io,
        eval_report_scaffolding,
        eval_report_update_payloads,
        eval_report_updates,
        eval_reports,
    )

    assert (
        eval_reports._authoritative_manifest_policy_selection
        is eval_policy_manifest_selection._authoritative_manifest_policy_selection
    )
    assert eval_reports._effective_manifest_git_commit is eval_policy_manifest_selection._effective_manifest_git_commit
    assert (
        eval_reports._persist_policy_selection_in_manifest
        is eval_policy_manifest_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._policy_selection_mode is eval_policy_manifest_selection._policy_selection_mode
    assert (
        eval_reports._resolve_selection_inputs_from_manifest
        is eval_policy_manifest_selection._resolve_selection_inputs_from_manifest
    )
    assert (
        eval_reports._run_summary_marks_canonical_eval_completed
        is eval_policy_manifest_selection._run_summary_marks_canonical_eval_completed
    )
    assert eval_reports._default_dev_eval_summaries_path is (
        eval_policy_final_set_resolution._default_dev_eval_summaries_path
    )
    assert eval_reports._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_reports._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports.RunLevelReportUpdateInputs is eval_report_update_payloads.RunLevelReportUpdateInputs
    assert eval_reports.build_run_summary_update_fields is eval_report_update_payloads.build_run_summary_update_fields
    assert (
        eval_reports.build_determinism_report_update_fields
        is eval_report_update_payloads.build_determinism_report_update_fields
    )
    assert eval_reports._load_json_object is eval_report_io._load_json_object
    assert eval_reports._expected_sha256 is eval_report_io._expected_sha256
    assert eval_reports._load_run_summary_or_default is eval_report_scaffolding._load_run_summary_or_default
    assert eval_reports._ensure_run_level_report_scaffolding is (
        eval_report_scaffolding._ensure_run_level_report_scaffolding
    )
    assert eval_reports._resolve_policy_ids_for_run is eval_policy_selection._resolve_policy_ids_for_run
    assert eval_policy_selection._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_policy_selection._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports._persist_policy_selection_in_manifest is (
        eval_policy_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._update_run_level_reports is eval_report_updates._update_run_level_reports


def test_eval_policy_selection_results_build_explicit_cli_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _explicit_policy_selection

    assert _explicit_policy_selection([" B0 RandomLegal ", "", "policy_000100"]) == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "explicit_cli", "policy_count": 2},
    )
    assert _explicit_policy_selection(["", "   "]) is None


def test_eval_policy_selection_results_build_manifest_fallback_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _manifest_policy_selection_fallback

    assert _manifest_policy_selection_fallback({"policy_set_selection": [" B0 RandomLegal ", "", 123]}) == (
        ["B0 RandomLegal", "123"],
        {"mode": "manifest_policy_set_selection_fallback", "policy_count": 2},
    )
    assert _manifest_policy_selection_fallback({"policy_set_selection": "not-a-list"}) is None
    assert _manifest_policy_selection_fallback({}) is None


def test_eval_policy_final_set_resolution_uses_available_source_paths(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_available_policy_source_paths

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    default_registry = layout.training_snapshots_dir / "registry.json"
    default_registry.write_text("{}\n", encoding="utf-8")
    periodic_dev_eval = layout.training_logs_dir / "periodic_dev_eval_summaries.json"
    periodic_dev_eval.write_text("{}\n", encoding="utf-8")
    manifest_registry = tmp_path / "manifest" / "registry.json"
    manifest_registry.parent.mkdir(parents=True, exist_ok=True)
    manifest_registry.write_text("{}\n", encoding="utf-8")
    explicit_dev_eval = tmp_path / "explicit" / "dev_eval.json"
    explicit_dev_eval.parent.mkdir(parents=True, exist_ok=True)
    explicit_dev_eval.write_text("{}\n", encoding="utf-8")

    resolved_registry, resolved_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=explicit_dev_eval,
        manifest_snapshot_registry=manifest_registry,
        manifest_dev_eval=None,
    )

    assert resolved_registry == manifest_registry
    assert resolved_dev_eval == explicit_dev_eval

    fallback_registry, fallback_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=tmp_path / "missing" / "registry.json",
        dev_eval_summaries_path=tmp_path / "missing" / "dev_eval.json",
        manifest_snapshot_registry=None,
        manifest_dev_eval=None,
    )

    assert fallback_registry is None
    assert fallback_dev_eval == periodic_dev_eval


def test_eval_policy_final_set_resolution_builds_deterministic_selection_details(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_deterministic_final_policy_set

    observed: dict[str, object] = {}
    registry_path = tmp_path / "registry.json"
    dev_eval_path = tmp_path / "dev_eval.json"

    def fake_resolve_final_policy_set(**kwargs: object) -> list[str]:
        observed["resolve"] = kwargs
        return ["B0 RandomLegal", "policy_000100"]

    resolved = _resolve_deterministic_final_policy_set(
        evaluation=SimpleNamespace(final_policy_set_selection={"folding": "seat_swap_mean"}, final_policy_set_size=2),
        resolved_snapshot_registry=registry_path,
        resolved_dev_eval=dev_eval_path,
        resolve_final_policy_set_fn=fake_resolve_final_policy_set,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {
            "mode": "deterministic_v1",
            "policy_count": 2,
            "snapshot_registry_path": registry_path.as_posix(),
            "dev_eval_summaries_path": dev_eval_path.as_posix(),
            "final_policy_set_size": 2,
        },
    )
    assert observed["resolve"] == {
        "snapshot_registry_path": registry_path,
        "dev_eval_summaries_path": dev_eval_path,
        "config": {"folding": "seat_swap_mean"},
        "final_policy_set_size": 2,
    }
    assert (
        _resolve_deterministic_final_policy_set(
            evaluation=SimpleNamespace(final_policy_set_selection={}, final_policy_set_size=2),
            resolved_snapshot_registry=None,
            resolved_dev_eval=dev_eval_path,
        )
        is None
    )


def test_eval_policy_final_set_resolution_reports_missing_inputs(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _raise_missing_final_policy_inputs

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match="requires a snapshot registry") as registry_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=None,
            resolved_dev_eval=tmp_path / "dev_eval.json",
            snapshot_registry_path=tmp_path / "explicit" / "registry.json",
            manifest_snapshot_registry=tmp_path / "manifest" / "registry.json",
            dev_eval_summaries_path=None,
            manifest_dev_eval=None,
        )
    assert str(tmp_path / "explicit" / "registry.json") in str(registry_exc.value)

    with pytest.raises(FileNotFoundError, match="requires dev-eval summaries") as dev_eval_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=tmp_path / "registry.json",
            resolved_dev_eval=None,
            snapshot_registry_path=None,
            manifest_snapshot_registry=None,
            dev_eval_summaries_path=tmp_path / "explicit" / "dev_eval.json",
            manifest_dev_eval=tmp_path / "manifest" / "dev_eval.json",
        )
    message = str(dev_eval_exc.value)
    assert (tmp_path / "explicit" / "dev_eval.json").as_posix() in message
    assert (tmp_path / "manifest" / "dev_eval.json").as_posix() in message
    assert "training/logs/dev_eval_summaries.json" in message
    assert "training/logs/periodic_dev_eval_summaries.json" in message


def test_eval_policy_manifest_selection_resolves_source_paths_from_manifest(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _resolve_selection_inputs_from_manifest

    absolute_dev_eval = tmp_path / "external" / "dev_eval.json"
    snapshot_registry, dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=tmp_path / "stack",
        manifest={
            "policy_set_selection_details": {
                "source_paths": {
                    "snapshot_registry_json": "runs/main/training/snapshots/registry.json",
                    "dev_eval_summaries_json": absolute_dev_eval.as_posix(),
                }
            }
        },
    )

    assert snapshot_registry == tmp_path / "stack" / "runs" / "main" / "training" / "snapshots" / "registry.json"
    assert dev_eval == absolute_dev_eval


def test_eval_policy_manifest_selection_requires_completed_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _authoritative_manifest_policy_selection

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": [" B0 RandomLegal ", "policy_000100"],
        "policy_set_selection_details": {"mode": "deterministic_v1", "status": "resolved"},
    }

    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
        )
        is None
    )

    layout.run_summary_path.write_text(
        json.dumps({"canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolved = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "deterministic_v1", "status": "resolved", "policy_count": 2},
    )
    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=tmp_path / "registry.json",
            dev_eval_summaries_path=None,
        )
        is None
    )


def test_eval_report_update_payloads_preserve_summary_and_determinism_fields(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_report_update_payloads import (
        RunLevelReportUpdateInputs,
        build_determinism_report_update_fields,
        build_run_summary_update_fields,
    )

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    selection_details = {"mode": "deterministic_v1", "status": "resolved"}
    inputs = RunLevelReportUpdateInputs(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details=selection_details,
        final_eval_payload={"matchups": [{"a": 1}, {"a": 2}]},
        metagame_payload={"kind": "metagame"},
        figure_paths=(layout.figures_paper_dir / "seat_bias.pdf", tmp_path / "external.pdf"),
        readiness_payload={"passed": True},
    )

    assert build_run_summary_update_fields(inputs) == {
        "final_eval_dir": "eval/final_eval",
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "policy_set_selection_mode": "deterministic_v1",
        "metagame_dir": "eval/metagame",
        "figure_outputs": ["figures/paper/seat_bias.pdf", (tmp_path / "external.pdf").as_posix()],
        "paper_readiness_summary_path": "paper_readiness_summary.json",
        "paper_grade": True,
        "canonical_eval_completed": True,
    }

    determinism_fields = build_determinism_report_update_fields(
        inputs,
        replay_verification={
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        artifact_hashes={"artifacts": {"summary.json": "ab" * 32}},
    )

    assert determinism_fields == {
        "run_dir": layout.run_dir.as_posix(),
        "policy_selection_mode": "deterministic_v1",
        "replay_verification": {
            "path": "eval/diagnostics/replay_verification.json",
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        "canonical_artifact_hashes": {"summary.json": "ab" * 32},
        "final_eval": {
            "path": "eval/final_eval/summary.json",
            "policy_ids": ["B0 RandomLegal", "policy_000100"],
            "selection": selection_details,
            "matchup_count": 2,
        },
    }


def test_eval_report_update_writes_summary_and_determinism_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_report_updates import _update_run_level_reports

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "preexisting": "summary"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.determinism_report_path.write_text(
        json.dumps({"kind": "determinism_report_v1", "preexisting": "determinism"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.replay_verification_json().write_text(
        json.dumps(
            {
                "status": "verified",
                "sampled_episode_count": 3,
                "verified_episode_count": 3,
                "failed_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps({"artifacts": {"summary.json": "cd" * 32}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _update_run_level_reports(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"mode": "deterministic_v1", "status": "resolved"},
        final_eval_payload={"matchups": [{"winner": "a"}]},
        metagame_payload=None,
        figure_paths=(),
        readiness_payload={"passed": False},
    )

    run_summary = json.loads(layout.run_summary_path.read_text(encoding="utf-8"))
    assert run_summary["preexisting"] == "summary"
    assert run_summary["final_eval_dir"] == "eval/final_eval"
    assert run_summary["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert run_summary["policy_set_selection_mode"] == "deterministic_v1"
    assert run_summary["metagame_dir"] is None
    assert run_summary["figure_outputs"] == []
    assert run_summary["paper_grade"] is False
    assert run_summary["canonical_eval_completed"] is True

    determinism = json.loads(layout.determinism_report_path.read_text(encoding="utf-8"))
    assert determinism["preexisting"] == "determinism"
    assert determinism["policy_selection_mode"] == "deterministic_v1"
    assert determinism["replay_verification"] == {
        "path": "eval/diagnostics/replay_verification.json",
        "status": "verified",
        "sampled_episode_count": 3,
        "verified_episode_count": 3,
        "failed_episode_count": 0,
    }
    assert determinism["canonical_artifact_hashes"] == {"summary.json": "cd" * 32}
    assert determinism["final_eval"]["matchup_count"] == 1


def test_eval_entrypoint_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    class FakeTensorBoardLogger:
        pass

    def fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(eval_script, "TensorBoardLogger", FakeTensorBoardLogger)
    monkeypatch.setattr(eval_script, "run_final_eval", fake_run_final_eval)

    dependencies = eval_script._canonical_eval_dependencies()

    assert dependencies.tensorboard_logger_cls is FakeTensorBoardLogger
    assert dependencies.run_final_eval_fn is fake_run_final_eval


def test_eval_dispatch_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    def fake_public_demo_eval_mode(**_kwargs: object) -> None:
        return None

    def fake_summary_json(_path: Path, _payload: object) -> None:
        return None

    monkeypatch.setattr(eval_script, "run_public_demo_eval_mode", fake_public_demo_eval_mode)
    monkeypatch.setattr(eval_script, "write_matchup_summary_json", fake_summary_json)

    dependencies = eval_script._eval_dispatch_dependencies()

    assert dependencies.run_public_demo_eval_mode_fn is fake_public_demo_eval_mode
    assert dependencies.write_matchup_summary_json_fn is fake_summary_json


def test_eval_startup_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    def fake_load_stack_config(_path: Path) -> object:
        return object()

    def fake_banner(
        _reported_spec_hash: str,
        _config_hash256: str,
        *,
        run_label: str,
        spec_mismatch_policy: str,
    ) -> None:
        assert run_label
        assert spec_mismatch_policy

    monkeypatch.setattr(eval_script, "load_stack_config", fake_load_stack_config)
    monkeypatch.setattr(eval_script, "print_startup_banner", fake_banner)

    dependencies = eval_script._eval_startup_dependencies()

    assert dependencies.load_stack_config_fn is fake_load_stack_config
    assert dependencies.print_startup_banner_fn is fake_banner


def test_eval_entrypoint_honors_completed_manifest_policy_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_locked"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_locked"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == ["B0 RandomLegal", "policy_locked"]
    assert details["mode"] == "deterministic_v1"
    assert details["status"] == "resolved"
    assert details["policy_count"] == 2
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_ignores_incomplete_manifest_selection_from_canonical_eval_pipeline(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_incomplete"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
            "resolved_by": "canonical_eval_pipeline_v1",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
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
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_ignores_completed_explicit_cli_manifest_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_explicit_cli"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.final_eval_summary_json().parent.mkdir(parents=True, exist_ok=True)
    layout.final_eval_summary_json().write_text("{}\n", encoding="utf-8")
    manifest = {
        "policy_set_selection": ["policy_custom_only"],
        "policy_set_selection_details": {
            "mode": "explicit_cli",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
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
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_manifest_persistence_records_explicit_cli_policy_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    run_dir = tmp_path / "runs" / "eval_manifest_persistence"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": ["policy_original"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }
    layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eval_script._persist_policy_selection_in_manifest(
        layout=layout,
        manifest=dict(manifest),
        policy_ids=["policy_explicit"],
        selection_details={"mode": "explicit_cli", "policy_count": 1},
    )

    persisted = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert persisted["policy_set_selection"] == ["policy_explicit"]
    assert persisted["policy_set_selection_details"] == {
        "mode": "explicit_cli",
        "policy_count": 1,
        "resolved_by": "canonical_eval_pipeline_v1",
        "status": "resolved",
    }


def test_eval_report_helpers_create_defaults_for_interpolated_runs(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    run_dir = tmp_path / "runs" / "interpolated_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "run_id64": "ab" * 8,
                "evaluation_pinning": {"eval_device": "cpu"},
                "seed_derivation": {"base_seed": 7},
                "seed_files": {"report_eval": {"path": "seeds/report.txt", "sha256": "cd" * 32}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_summary = eval_script._load_run_summary_or_default(layout)
    determinism = eval_script._load_determinism_report_or_default(layout)
    environment = eval_script._load_environment_or_default(layout)

    assert run_summary["runtime_mode"] == "interpolated_checkpoint"
    assert run_summary["run_id256"] == "ab" * 32
    assert determinism["device_policy"]["learner"] == "interpolated_checkpoint"
    assert determinism["device_policy"]["evaluation"] == "cpu"
    assert determinism["seed_derivation"] == {"base_seed": 7}
    assert environment["kind"] == "environment_manifest_v1"
    assert environment["run_id256"] == "ab" * 32


def test_eval_pipeline_persists_policy_selection_before_run_final_eval(tmp_path: Path, monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    expected_policy_ids = [
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
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_pipeline_persist_before_final_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "config_hash256": "cd" * 32,
                "spec_hash256": "ef" * 32,
                "policy_set_selection": [],
                "policy_set_selection_details": {
                    "status": "unresolved",
                    "reason": "selection_pending",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeTensorBoardLogger:
        enabled = False

        def __init__(self, _log_dir: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    observed: dict[str, dict[str, object]] = {}

    def _fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        observed["manifest"] = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        raise RuntimeError("stop after manifest check")

    monkeypatch.setattr(eval_script, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(
        eval_script,
        "load_verified_simulator_contract",
        lambda *_args, **_kwargs: _FakeContract(),
    )
    monkeypatch.setattr(eval_script, "resolve_eval_policies", lambda **_kwargs: [])
    monkeypatch.setattr(eval_script, "SimulatorEvalRunner", lambda **_kwargs: object())
    monkeypatch.setattr(eval_script, "run_final_eval", _fake_run_final_eval)

    try:
        eval_script._run_canonical_eval_pipeline(
            parser=eval_script.argparse.ArgumentParser(),
            stack=stack,
            run_dir=run_dir,
            final_eval_dir=None,
            policy_ids=[],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            b1_baseline_run_dir=None,
            bootstrap_samples=8,
            paired_seed_limit=1,
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            skip_metagame=True,
            study_config_path=None,
            skip_figures=True,
            skip_readiness=True,
            git_commit_override="",
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after manifest check"
    else:
        raise AssertionError("expected fake run_final_eval to stop the pipeline")

    persisted = observed["manifest"]
    assert persisted["policy_set_selection"] == expected_policy_ids
    assert persisted["policy_set_selection_details"] == {
        "mode": "deterministic_v1",
        "policy_count": len(expected_policy_ids),
        "resolved_by": "canonical_eval_pipeline_v1",
        "snapshot_registry_path": (layout.training_snapshots_dir / "registry.json").as_posix(),
        "dev_eval_summaries_path": (layout.training_logs_dir / "periodic_dev_eval_summaries.json").as_posix(),
        "final_policy_set_size": 10,
        "status": "resolved",
    }


def test_eval_git_commit_override_does_not_mutate_manifest_payload() -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    manifest = {"run_id256": "ab" * 32}

    effective = eval_script._effective_manifest_git_commit(
        manifest=manifest,
        git_commit_override="deadbeef" * 5,
    )

    assert effective == "deadbeef" * 5
    assert "git_commit" not in manifest
