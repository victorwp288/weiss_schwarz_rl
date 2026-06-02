from __future__ import annotations

from .test_entrypoints import (
    Path,
    argparse,
    pytest,
)


def test_eval_entrypoint_main_runner_threads_startup_and_dispatch() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_entrypoint_support.main import run_eval_entrypoint_main

    calls: list[str] = []
    parsed_args = SimpleNamespace(kind="args")
    validated_state = SimpleNamespace(run_label="eval_label")
    startup_state = SimpleNamespace(kind="startup")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    class FakeParser:
        def parse_args(self) -> object:
            calls.append("parse_args")
            return parsed_args

    parser_obj = FakeParser()

    def fake_build_parser() -> FakeParser:
        calls.append("build_parser")
        return parser_obj

    def fake_validate(*, parser: object, args: object) -> object:
        calls.append("validate")
        assert parser is parser_obj
        assert args is parsed_args
        return validated_state

    def fake_startup_dependencies() -> object:
        calls.append("startup_dependencies")
        return startup_dependencies

    def fake_prepare_startup(*, args: object, run_label: str, dependencies: object) -> object:
        calls.append("prepare_startup")
        assert args is parsed_args
        assert run_label == "eval_label"
        assert dependencies is startup_dependencies
        return startup_state

    def fake_dispatch_dependencies() -> object:
        calls.append("dispatch_dependencies")
        return dispatch_dependencies

    def fake_dispatch(
        *,
        parser: object,
        args: object,
        validated: object,
        startup: object,
        dependencies: object,
    ) -> None:
        calls.append("dispatch")
        assert parser is parser_obj
        assert args is parsed_args
        assert validated is validated_state
        assert startup is startup_state
        assert dependencies is dispatch_dependencies

    run_eval_entrypoint_main(
        build_eval_parser_fn=fake_build_parser,
        validate_eval_args_fn=fake_validate,
        prepare_eval_startup_fn=fake_prepare_startup,
        run_eval_dispatch_fn=fake_dispatch,
        eval_startup_dependencies_fn=fake_startup_dependencies,
        eval_dispatch_dependencies_fn=fake_dispatch_dependencies,
    )

    assert calls == [
        "build_parser",
        "parse_args",
        "validate",
        "startup_dependencies",
        "prepare_startup",
        "dispatch_dependencies",
        "dispatch",
    ]


def test_eval_entrypoint_runtime_threads_facade_globals(monkeypatch) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows import eval_entrypoint_runtime

    calls: list[str] = []
    parser_obj = SimpleNamespace(kind="parser")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    def fake_main(**kwargs: object) -> None:
        calls.append("main")
        assert kwargs["build_eval_parser_fn"]() is parser_obj
        assert kwargs["eval_startup_dependencies_fn"]() is startup_dependencies
        assert kwargs["eval_dispatch_dependencies_fn"]() is dispatch_dependencies

    def fake_startup_dependencies(entrypoint_globals: object) -> object:
        calls.append("startup_dependencies")
        assert entrypoint_globals is globals_map
        return startup_dependencies

    def fake_dispatch_dependencies(entrypoint_globals: object) -> object:
        calls.append("dispatch_dependencies")
        assert entrypoint_globals is globals_map
        return dispatch_dependencies

    monkeypatch.setattr(eval_entrypoint_runtime, "run_eval_entrypoint_main", fake_main)
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_startup_dependencies",
        fake_startup_dependencies,
    )
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_dispatch_dependencies",
        fake_dispatch_dependencies,
    )
    globals_map = {
        "build_eval_parser": lambda: parser_obj,
        "validate_eval_args": object(),
        "prepare_eval_startup": object(),
        "run_eval_dispatch": object(),
    }

    eval_entrypoint_runtime.run_eval_entrypoint(entrypoint_globals=globals_map)

    assert calls == ["main", "startup_dependencies", "dispatch_dependencies"]


def test_eval_entrypoint_runtime_canonical_wrapper_uses_facade_globals(monkeypatch, tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows import eval_entrypoint_runtime

    observed: dict[str, object] = {}
    dependencies = object()
    pipeline = object()
    entrypoint_adapter = object()

    def fake_dependencies(entrypoint_globals: object) -> object:
        observed["dependency_globals"] = entrypoint_globals
        return dependencies

    def fake_wrapper(**kwargs: object) -> int:
        observed["wrapper"] = kwargs
        assert kwargs["canonical_dependencies_fn"]() is dependencies
        return 41

    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_canonical_dependencies",
        fake_dependencies,
    )
    monkeypatch.setattr(eval_entrypoint_runtime, "run_entrypoint_canonical_eval_pipeline", fake_wrapper)
    globals_map = {
        "run_canonical_eval_pipeline": pipeline,
        "run_canonical_eval_entrypoint_pipeline": entrypoint_adapter,
    }
    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")

    result = eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline(
        entrypoint_globals=globals_map,
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
    )

    assert result == 41
    assert observed["dependency_globals"] is globals_map
    wrapper_call = observed["wrapper"]
    assert wrapper_call["parser"] is parser
    assert wrapper_call["stack"] is stack
    assert wrapper_call["run_canonical_eval_pipeline_fn"] is pipeline
    assert wrapper_call["run_canonical_eval_entrypoint_pipeline_fn"] is entrypoint_adapter


def test_eval_entrypoint_compat_canonical_wrapper_forwards_callables(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_entrypoint_support.compat import run_entrypoint_canonical_eval_pipeline

    observed: dict[str, object] = {}
    dependencies = object()

    def fake_dependencies() -> object:
        observed["dependencies_called"] = True
        return dependencies

    def fake_pipeline(**kwargs: object) -> int:
        observed["pipeline"] = kwargs
        return 29

    def fake_entrypoint_adapter(**kwargs: object) -> int:
        observed["adapter"] = kwargs
        return kwargs["run_canonical_eval_pipeline_fn"](
            parser=kwargs["parser"],
            stack=kwargs["stack"],
            run_dir=kwargs["run_dir"],
            final_eval_dir=kwargs["final_eval_dir"],
            policy_ids=kwargs["policy_ids"],
            snapshot_registry_path=kwargs["snapshot_registry_path"],
            dev_eval_summaries_path=kwargs["dev_eval_summaries_path"],
            b1_baseline_run_dir=kwargs["b1_baseline_run_dir"],
            bootstrap_samples=kwargs["bootstrap_samples"],
            paired_seed_limit=kwargs["paired_seed_limit"],
            stage1_paired_seeds=kwargs["stage1_paired_seeds"],
            max_paired_seeds=kwargs["max_paired_seeds"],
            skip_metagame=kwargs["skip_metagame"],
            study_config_path=kwargs["study_config_path"],
            skip_figures=kwargs["skip_figures"],
            skip_readiness=kwargs["skip_readiness"],
            git_commit_override=kwargs["git_commit_override"],
            dependencies=kwargs["canonical_dependencies_fn"](),
        )

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")

    result = run_entrypoint_canonical_eval_pipeline(
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        canonical_dependencies_fn=fake_dependencies,
        run_canonical_eval_pipeline_fn=fake_pipeline,
        run_canonical_eval_entrypoint_pipeline_fn=fake_entrypoint_adapter,
    )

    assert result == 29
    assert observed["dependencies_called"] is True
    adapter_call = observed["adapter"]
    assert adapter_call["canonical_dependencies_fn"] is fake_dependencies
    assert adapter_call["run_canonical_eval_pipeline_fn"] is fake_pipeline
    pipeline_call = observed["pipeline"]
    assert pipeline_call["parser"] is parser
    assert pipeline_call["stack"] is stack
    assert pipeline_call["run_dir"] == tmp_path / "run"
    assert pipeline_call["final_eval_dir"] == tmp_path / "final"
    assert pipeline_call["policy_ids"] == ["B0 RandomLegal"]
    assert pipeline_call["snapshot_registry_path"] == tmp_path / "registry.json"
    assert pipeline_call["dev_eval_summaries_path"] == tmp_path / "dev.json"
    assert pipeline_call["b1_baseline_run_dir"] == tmp_path / "b1"
    assert pipeline_call["bootstrap_samples"] == 8
    assert pipeline_call["paired_seed_limit"] == 1
    assert pipeline_call["stage1_paired_seeds"] == 2
    assert pipeline_call["max_paired_seeds"] == 3
    assert pipeline_call["skip_metagame"] is True
    assert pipeline_call["study_config_path"] == tmp_path / "study.yaml"
    assert pipeline_call["skip_figures"] is True
    assert pipeline_call["skip_readiness"] is True
    assert pipeline_call["git_commit_override"] == "abc123"
    assert pipeline_call["dependencies"] is dependencies


def test_canonical_eval_entrypoint_request_preserves_flat_kwargs(tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.entrypoint_request import (
        canonical_eval_entrypoint_request,
        run_canonical_entrypoint_request_adapter,
        run_canonical_entrypoint_request_pipeline,
    )

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")
    dependencies = object()
    request = canonical_eval_entrypoint_request(
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=("B0 RandomLegal", "policy_000100"),
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples="8",
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=1,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=0,
        skip_readiness=True,
        git_commit_override=123,
    )

    entrypoint_kwargs = request.entrypoint_kwargs()
    pipeline_kwargs = request.pipeline_kwargs(dependencies=dependencies)

    assert request.policy_ids == ["B0 RandomLegal", "policy_000100"]
    assert request.bootstrap_samples == 8
    assert request.skip_metagame is True
    assert request.skip_figures is False
    assert request.git_commit_override == "123"
    assert entrypoint_kwargs["parser"] is parser
    assert entrypoint_kwargs["stack"] is stack
    assert entrypoint_kwargs["run_dir"] == tmp_path / "run"
    assert entrypoint_kwargs["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert "dependencies" not in entrypoint_kwargs
    assert pipeline_kwargs["dependencies"] is dependencies

    observed: dict[str, object] = {}

    def fake_pipeline(**kwargs: object) -> int:
        observed["pipeline"] = kwargs
        return 19

    def fake_adapter(**kwargs: object) -> int:
        observed["adapter"] = kwargs
        return 23

    assert (
        run_canonical_entrypoint_request_pipeline(
            request=request,
            dependencies=dependencies,
            run_canonical_eval_pipeline_fn=fake_pipeline,
        )
        == 19
    )
    assert observed["pipeline"] == pipeline_kwargs
    assert (
        run_canonical_entrypoint_request_adapter(
            request=request,
            canonical_dependencies_fn=lambda: dependencies,
            run_canonical_eval_pipeline_fn=fake_pipeline,
            run_canonical_eval_entrypoint_pipeline_fn=fake_adapter,
        )
        == 23
    )
    adapter_call = observed["adapter"]
    assert adapter_call["parser"] is parser
    assert adapter_call["canonical_dependencies_fn"]() is dependencies
    assert adapter_call["run_canonical_eval_pipeline_fn"] is fake_pipeline


def test_canonical_eval_entrypoint_adapter_injects_dependencies(tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.entrypoint_adapter import run_canonical_eval_entrypoint_pipeline

    observed: dict[str, object] = {}
    dependencies = object()

    def fake_pipeline(**kwargs: object) -> int:
        observed.update(kwargs)
        return 17

    result = run_canonical_eval_entrypoint_pipeline(
        parser=argparse.ArgumentParser(),
        stack=SimpleNamespace(name="stack"),
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        canonical_dependencies_fn=lambda: dependencies,
        run_canonical_eval_pipeline_fn=fake_pipeline,
    )

    assert result == 17
    assert observed["run_dir"] == tmp_path / "run"
    assert observed["final_eval_dir"] == tmp_path / "final"
    assert observed["policy_ids"] == ["B0 RandomLegal"]
    assert observed["snapshot_registry_path"] == tmp_path / "registry.json"
    assert observed["dev_eval_summaries_path"] == tmp_path / "dev.json"
    assert observed["b1_baseline_run_dir"] == tmp_path / "b1"
    assert observed["bootstrap_samples"] == 8
    assert observed["paired_seed_limit"] == 1
    assert observed["stage1_paired_seeds"] == 2
    assert observed["max_paired_seeds"] == 3
    assert observed["skip_metagame"] is True
    assert observed["study_config_path"] == tmp_path / "study.yaml"
    assert observed["skip_figures"] is True
    assert observed["skip_readiness"] is True
    assert observed["git_commit_override"] == "abc123"
    assert observed["dependencies"] is dependencies


def test_canonical_eval_runtime_phase_persists_selection_before_loading_policies(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.phases import (
        CanonicalEvalRunState,
        resolve_canonical_eval_runtime_state,
    )

    calls: list[str] = []
    observed: dict[str, object] = {}
    layout = SimpleNamespace()
    evaluation = SimpleNamespace(
        eval_assert_sorted_legal_ids=True,
        replay_capture_rate_eval=0.25,
        regression_capture_count=2,
        final_matrix_stage1_paired_seeds=2,
        final_matrix_stage2_adaptive_max_paired_seeds=3,
    )
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"report_eval": tmp_path / "report_eval_seeds.txt"},
        config=SimpleNamespace(evaluation=evaluation),
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=evaluation,
        study_config=None,
    )

    class FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    def fake_resolve_policy_ids_for_run_fn(**_kwargs: object) -> tuple[list[str], dict[str, object], None, None]:
        calls.append("resolve_policy_ids")
        return ["B0 RandomLegal", "policy_000100"], {"status": "resolved"}, None, None

    def fake_persist_policy_selection_in_manifest_fn(**kwargs: object) -> None:
        calls.append("persist_selection")
        observed["persisted"] = kwargs

    def fake_load_verified_simulator_contract_fn(*_args: object, **_kwargs: object) -> FakeContract:
        calls.append("load_contract")
        return FakeContract()

    def fake_resolve_eval_policies_fn(**kwargs: object) -> list[str]:
        calls.append("resolve_eval_policies")
        observed["policy_resolution"] = kwargs
        return ["policy-object"]

    def fake_simulator_eval_runner_cls(**kwargs: object) -> object:
        calls.append("build_runner")
        observed["runner"] = kwargs
        return object()

    def fake_parse_seed_file_fn(path: Path) -> list[int]:
        calls.append("parse_seeds")
        observed["seed_file"] = path
        return [101, 202, 303]

    dependencies = SimpleNamespace(
        resolve_policy_ids_for_run_fn=fake_resolve_policy_ids_for_run_fn,
        persist_policy_selection_in_manifest_fn=fake_persist_policy_selection_in_manifest_fn,
        load_verified_simulator_contract_fn=fake_load_verified_simulator_contract_fn,
        resolve_eval_policies_fn=fake_resolve_eval_policies_fn,
        simulator_eval_runner_cls=fake_simulator_eval_runner_cls,
        parse_seed_file_fn=fake_parse_seed_file_fn,
    )

    runtime_state = resolve_canonical_eval_runtime_state(
        stack=stack,
        run_dir=tmp_path / "run",
        policy_ids=[],
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        b1_baseline_run_dir=tmp_path / "b1",
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        run_state=run_state,
        dependencies=dependencies,
    )

    assert calls == [
        "resolve_policy_ids",
        "persist_selection",
        "load_contract",
        "resolve_eval_policies",
        "build_runner",
        "parse_seeds",
    ]
    assert runtime_state.policy_ids == ["B0 RandomLegal", "policy_000100"]
    assert runtime_state.paired_seeds == [101, 202]
    assert runtime_state.paired_seed_limit == 2
    assert runtime_state.stage1_paired_seeds == 2
    assert runtime_state.max_paired_seeds == 2
    assert observed["policy_resolution"]["observation_dim"] == 512
    assert observed["policy_resolution"]["action_dim"] == 9
    assert observed["policy_resolution"]["b1_baseline_run_dir"] == tmp_path / "b1"
    assert observed["runner"]["pass_action_id"] == 8
    assert observed["runner"]["require_sorted_legal_ids"] is True


def test_canonical_eval_seed_budget_preserves_limit_defaults_and_errors(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.seed_budget import resolve_canonical_eval_seed_budget

    seed_path = tmp_path / "report_eval_seeds.txt"
    stack = SimpleNamespace(seed_sets={"report_eval": seed_path})
    evaluation = SimpleNamespace(
        final_matrix_stage1_paired_seeds=4,
        final_matrix_stage2_adaptive_max_paired_seeds=8,
    )
    dependencies = SimpleNamespace(parse_seed_file_fn=lambda path: [11, 22, 33] if path == seed_path else [])

    seed_budget = resolve_canonical_eval_seed_budget(
        stack=stack,
        evaluation=evaluation,
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        dependencies=dependencies,
    )

    assert seed_budget.seed_file_path == seed_path
    assert seed_budget.paired_seeds == [11, 22]
    assert seed_budget.paired_seed_limit == 2
    assert seed_budget.stage1_paired_seeds == 2
    assert seed_budget.max_paired_seeds == 2

    with pytest.raises(ValueError, match=r"stage1 paired seeds \(3\) cannot exceed max paired seeds \(2\)"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=3,
            max_paired_seeds=2,
            dependencies=dependencies,
        )

    empty_dependencies = SimpleNamespace(parse_seed_file_fn=lambda _path: [])
    with pytest.raises(ValueError, match="report_eval seed file produced no usable seeds"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
            dependencies=empty_dependencies,
        )


def test_canonical_eval_output_phase_preserves_final_eval_metadata(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.phases import (
        CanonicalEvalRunState,
        CanonicalEvalRuntimeState,
        write_canonical_eval_outputs,
    )

    class FakeLayout:
        final_eval_dir = tmp_path / "run" / "eval" / "final_eval"
        metagame_dir = tmp_path / "run" / "eval" / "metagame"
        figures_paper_dir = tmp_path / "run" / "figures" / "paper"
        paper_readiness_summary_path = tmp_path / "run" / "paper_readiness_summary.json"

        def final_eval_summary_json(self) -> Path:
            return self.final_eval_dir / "summary.json"

        def replay_verification_json(self) -> Path:
            return self.final_eval_dir / "replay_verification.json"

    tensorboard_logger = SimpleNamespace(enabled=False)
    evaluation = SimpleNamespace(
        stop_rules={"minimum": 1},
        final_policy_set_selection=SimpleNamespace(folding="seat_swap_mean"),
        final_policy_set_size=2,
    )
    run_state = CanonicalEvalRunState(
        layout=FakeLayout(),
        tensorboard_logger=tensorboard_logger,
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=evaluation,
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev_eval.json",
        runner=object(),
        paired_seeds=[101, 202],
        paired_seed_limit=2,
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        seed_file_path=tmp_path / "report_eval_seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    observed: dict[str, object] = {}

    def fake_run_final_eval_fn(**kwargs: object) -> dict[str, object]:
        observed["final_eval"] = kwargs
        return {"kind": "summary"}

    dependencies = SimpleNamespace(
        tensorboard_unavailable_reason_fn=lambda: "no writer",
        run_final_eval_fn=fake_run_final_eval_fn,
        ensure_run_level_report_scaffolding_fn=lambda layout: observed.setdefault("scaffold", layout),
        update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs),
    )

    result = write_canonical_eval_outputs(
        run_dir=tmp_path / "run",
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=True,
        skip_readiness=True,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )

    assert result == 0
    final_eval_call = observed["final_eval"]
    assert final_eval_call["paired_seeds"] == [101, 202]
    assert final_eval_call["stage1_paired_seeds"] == 1
    assert final_eval_call["max_paired_seeds"] == 2
    assert final_eval_call["sample_count"] == 8
    assert final_eval_call["metadata"]["pipeline"] == {
        "kind": "canonical_eval_pipeline_v1",
        "selection": {"status": "resolved"},
        "seed_file": (tmp_path / "report_eval_seeds.txt").as_posix(),
        "paired_seed_limit": 2,
    }
    assert final_eval_call["metadata"]["recommended_focal_policy_id"] == "policy_000100"
    assert observed["reports"]["final_eval_payload"] == {"kind": "summary"}
    assert observed["reports"]["metagame_payload"] is None
    assert observed["reports"]["figure_paths"] == ()
    assert observed["reports"]["readiness_payload"] is None
    assert "Resolved policy set: ['B0 RandomLegal', 'policy_000100']" in capsys.readouterr().out


def test_canonical_supplemental_outputs_builds_thesis_artifacts_in_order(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import build_canonical_supplemental_outputs

    calls: list[str] = []
    observed: dict[str, object] = {}
    layout = SimpleNamespace(
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=SimpleNamespace(metagame={"m": 1}, sensitivity={"s": 2}),
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_metagame(**kwargs: object) -> dict[str, object]:
        calls.append("metagame")
        observed["metagame"] = kwargs
        return {"metagame": "payload"}

    def fake_figures(run_dir: Path) -> tuple[Path, ...]:
        calls.append("figures")
        observed["figures"] = run_dir
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    def fake_scaffold(scaffold_layout: object) -> None:
        calls.append("scaffold")
        observed["scaffold"] = scaffold_layout

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        calls.append("readiness")
        observed["readiness"] = kwargs
        return {"passed": True}

    def fake_write_readiness(path: Path, payload: dict[str, object]) -> None:
        calls.append("write_readiness")
        observed["write_readiness"] = (path, payload)

    dependencies = SimpleNamespace(
        build_sensitivity_report_fn=fake_metagame,
        render_paper_figures_fn=fake_figures,
        ensure_run_level_report_scaffolding_fn=fake_scaffold,
        build_paper_readiness_summary_fn=fake_readiness,
        write_paper_readiness_json_fn=fake_write_readiness,
    )

    outputs = build_canonical_supplemental_outputs(
        run_dir=tmp_path / "run",
        skip_metagame=False,
        skip_figures=False,
        skip_readiness=False,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )

    assert calls == ["metagame", "figures", "scaffold", "readiness", "write_readiness"]
    assert outputs.metagame_payload == {"metagame": "payload"}
    assert outputs.figure_paths == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)
    assert outputs.readiness_payload == {"passed": True}
    assert observed["metagame"] == {
        "final_eval_dir": layout.final_eval_dir,
        "out_dir": layout.metagame_dir,
        "metagame_config": {"m": 1},
        "sensitivity_config": {"s": 2},
    }
    assert observed["figures"] == tmp_path / "run"
    assert observed["scaffold"] is layout
    assert observed["readiness"] == {"run_dir": tmp_path / "run", "focal_policy_id": "policy_000100"}
    assert observed["write_readiness"] == (layout.paper_readiness_summary_path, {"passed": True})


def test_canonical_output_bundle_builds_final_eval_before_supplemental_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import weiss_rl.workflows.canonical_eval.output_bundle as output_bundle_module
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    calls: list[str] = []
    run_state = CanonicalEvalRunState(
        layout=SimpleNamespace(),
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_final_eval(**kwargs: object) -> dict[str, object]:
        calls.append("final")
        assert kwargs["bootstrap_samples"] == 8
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return {"final": "payload"}

    def fake_supplemental(**kwargs: object) -> CanonicalEvalSupplementalOutputs:
        calls.append("supplemental")
        assert kwargs["run_dir"] == tmp_path / "run"
        assert kwargs["skip_metagame"] is True
        assert kwargs["skip_figures"] is False
        assert kwargs["skip_readiness"] is True
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
            readiness_payload=None,
        )

    monkeypatch.setattr(output_bundle_module, "run_canonical_final_eval_output", fake_final_eval)
    monkeypatch.setattr(output_bundle_module, "build_canonical_supplemental_outputs", fake_supplemental)

    bundle = output_bundle_module.build_canonical_eval_output_bundle(
        run_dir=tmp_path / "run",
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=False,
        skip_readiness=True,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(),
    )

    assert calls == ["final", "supplemental"]
    assert bundle.final_eval_payload == {"final": "payload"}
    assert bundle.supplemental.figure_paths == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)


def test_canonical_metagame_output_forwards_study_configs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.metagame_outputs import build_canonical_metagame_output

    observed: dict[str, object] = {}
    layout = SimpleNamespace(
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
    )
    study_config = SimpleNamespace(metagame={"m": 1}, sensitivity={"s": 2})

    def fake_metagame(**kwargs: object) -> dict[str, object]:
        observed["metagame"] = kwargs
        return {"metagame": "payload"}

    payload = build_canonical_metagame_output(
        layout=layout,
        study_config=study_config,
        dependencies=SimpleNamespace(build_sensitivity_report_fn=fake_metagame),
    )

    assert payload == {"metagame": "payload"}
    assert observed["metagame"] == {
        "final_eval_dir": layout.final_eval_dir,
        "out_dir": layout.metagame_dir,
        "metagame_config": {"m": 1},
        "sensitivity_config": {"s": 2},
    }


def test_canonical_figure_outputs_forward_run_dir(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.figure_outputs import build_canonical_figure_outputs

    observed: dict[str, object] = {}

    def fake_figures(run_dir: Path) -> tuple[Path, ...]:
        observed["run_dir"] = run_dir
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    outputs = build_canonical_figure_outputs(
        run_dir=tmp_path / "run",
        dependencies=SimpleNamespace(render_paper_figures_fn=fake_figures),
    )

    assert observed["run_dir"] == tmp_path / "run"
    assert outputs == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)


def test_canonical_readiness_output_writes_focal_policy_summary(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.readiness_outputs import build_canonical_readiness_output
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState

    observed: dict[str, object] = {}
    layout = SimpleNamespace(paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json")
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        observed["readiness"] = kwargs
        return {"passed": True, "focal_policy_id": kwargs["focal_policy_id"]}

    def fake_write(path: Path, payload: dict[str, object]) -> None:
        observed["write"] = (path, payload)

    payload = build_canonical_readiness_output(
        run_dir=tmp_path / "run",
        layout=layout,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(
            build_paper_readiness_summary_fn=fake_readiness,
            write_paper_readiness_json_fn=fake_write,
        ),
    )

    assert payload == {"passed": True, "focal_policy_id": "policy_000100"}
    assert observed["readiness"] == {"run_dir": tmp_path / "run", "focal_policy_id": "policy_000100"}
    assert observed["write"] == (layout.paper_readiness_summary_path, payload)


def test_canonical_output_message_renderer_handles_optional_outputs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.cli_messages import render_canonical_eval_output_messages
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    layout = SimpleNamespace(
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        figures_paper_dir=tmp_path / "run" / "figures" / "paper",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
        final_eval_summary_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "summary.json",
        replay_verification_json=lambda: tmp_path / "run" / "eval" / "diagnostics" / "replay_verification.json",
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    minimal_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(),
            readiness_payload=None,
        ),
    )

    assert minimal_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )

    full_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload={"kind": "summary"},
            figure_paths=(
                tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",
                tmp_path / "run" / "figures" / "paper" / "main_eval.png",
            ),
            readiness_payload={"passed": False},
        ),
    )

    assert full_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}",
        f"Rendered 2 paper figure files to {layout.figures_paper_dir}",
        f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}",
        "Paper readiness: failed",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )


def test_canonical_tensorboard_publication_handles_enabled_and_disabled(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs
    from weiss_rl.workflows.canonical_eval.tensorboard_publication import (
        begin_canonical_eval_tensorboard_logging,
        publish_canonical_eval_tensorboard_summaries,
    )

    class FakeTensorBoardLogger:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled
            self.calls: list[tuple[str, object]] = []

        def log_text(self, tag: str, payload: object) -> None:
            self.calls.append(("text", (tag, payload)))

        def log_final_eval_summary(self, payload: object, *, step: int) -> None:
            self.calls.append(("final", (payload, step)))

        def log_metagame_summary(self, payload: object, *, metagame_dir: Path, step: int) -> None:
            self.calls.append(("metagame", (payload, metagame_dir, step)))

        def log_paper_readiness(self, payload: object, *, step: int) -> None:
            self.calls.append(("readiness", (payload, step)))

    layout = SimpleNamespace(metagame_dir=tmp_path / "run" / "eval" / "metagame")
    enabled_logger = FakeTensorBoardLogger(enabled=True)
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=enabled_logger,
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(),
        readiness_payload={"passed": True},
    )

    begin_canonical_eval_tensorboard_logging(
        run_state=run_state,
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=enabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert enabled_logger.calls == [
        ("text", ("eval/run/manifest", {"run_id256": "ab" * 32})),
        ("final", ({"summary": "payload"}, 0)),
        ("metagame", ({"meta": "payload"}, layout.metagame_dir, 0)),
        ("readiness", ({"passed": True}, 0)),
    ]

    disabled_logger = FakeTensorBoardLogger(enabled=False)
    begin_canonical_eval_tensorboard_logging(
        run_state=CanonicalEvalRunState(
            layout=layout,
            tensorboard_logger=disabled_logger,
            manifest={},
            run_id256="",
            evaluation=SimpleNamespace(),
            study_config=None,
        ),
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=disabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert disabled_logger.calls == []
    assert "TensorBoard logging is disabled for eval: SummaryWriter unavailable" in capsys.readouterr().err


def test_canonical_report_publication_updates_run_level_reports(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.report_publication import publish_canonical_eval_run_reports
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    layout = SimpleNamespace()
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
        readiness_payload={"passed": True},
    )
    observed: dict[str, object] = {}

    publish_canonical_eval_run_reports(
        run_dir=tmp_path / "run",
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
        dependencies=SimpleNamespace(
            update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs)
        ),
    )

    assert observed["reports"] == {
        "layout": layout,
        "run_dir": tmp_path / "run",
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "selection_details": {"status": "resolved"},
        "final_eval_payload": {"summary": "payload"},
        "metagame_payload": {"meta": "payload"},
        "figure_paths": supplemental.figure_paths,
        "readiness_payload": {"passed": True},
    }


def test_canonical_output_publisher_updates_reports_tensorboard_and_cli(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.publisher import (
        begin_canonical_eval_output_logging,
        publish_canonical_eval_outputs,
    )
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    class FakeTensorBoardLogger:
        enabled = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def log_text(self, tag: str, payload: object) -> None:
            self.calls.append(("text", (tag, payload)))

        def log_final_eval_summary(self, payload: object, *, step: int) -> None:
            self.calls.append(("final", (payload, step)))

        def log_metagame_summary(self, payload: object, *, metagame_dir: Path, step: int) -> None:
            self.calls.append(("metagame", (payload, metagame_dir, step)))

        def log_paper_readiness(self, payload: object, *, step: int) -> None:
            self.calls.append(("readiness", (payload, step)))

    tensorboard_logger = FakeTensorBoardLogger()
    layout = SimpleNamespace(
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        figures_paper_dir=tmp_path / "run" / "figures" / "paper",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
        final_eval_summary_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "summary.json",
        replay_verification_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "replay_verification.json",
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=tensorboard_logger,
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
        readiness_payload={"passed": True},
    )
    observed: dict[str, object] = {}
    dependencies = SimpleNamespace(
        tensorboard_unavailable_reason_fn=lambda: None,
        update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs),
    )
    final_eval_payload = {"summary": "payload"}

    begin_canonical_eval_output_logging(run_state=run_state, dependencies=dependencies)
    publish_canonical_eval_outputs(
        run_dir=tmp_path / "run",
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload=final_eval_payload,
        supplemental=supplemental,
        dependencies=dependencies,
    )

    assert observed["reports"]["final_eval_payload"] is final_eval_payload
    assert observed["reports"]["metagame_payload"] == {"meta": "payload"}
    assert observed["reports"]["figure_paths"] == supplemental.figure_paths
    assert observed["reports"]["readiness_payload"] == {"passed": True}
    assert tensorboard_logger.calls == [
        ("text", ("eval/run/manifest", {"run_id256": "ab" * 32})),
        ("final", (final_eval_payload, 0)),
        ("metagame", ({"meta": "payload"}, layout.metagame_dir, 0)),
        ("readiness", ({"passed": True}, 0)),
    ]
    output = capsys.readouterr().out
    assert f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}" in output
    assert f"Canonical replay verification JSON: {layout.replay_verification_json()}" in output
    assert f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}" in output
    assert f"Rendered 1 paper figure files to {layout.figures_paper_dir}" in output
    assert f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}" in output
    assert "Paper readiness: passed" in output
    assert "Resolved policy set: ['B0 RandomLegal', 'policy_000100']" in output
