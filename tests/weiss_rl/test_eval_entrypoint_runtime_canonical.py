from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def test_eval_entrypoint_runtime_canonical_wrapper_uses_facade_globals(monkeypatch, tmp_path: Path) -> None:
    from importlib.util import find_spec

    import weiss_rl.workflows.eval_entrypoint_support.runtime as eval_entrypoint_runtime

    observed: dict[str, object] = {}
    dependencies = object()
    pipeline = object()
    entrypoint_adapter = object()

    def fake_dependencies(entrypoint_globals: object) -> object:
        observed["dependency_globals"] = entrypoint_globals
        return dependencies

    def fake_adapter(**kwargs: object) -> int:
        observed["adapter"] = kwargs
        assert kwargs["canonical_dependencies_fn"]() is dependencies
        return 41

    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_canonical_dependencies",
        fake_dependencies,
    )
    monkeypatch.setattr(eval_entrypoint_runtime, "run_canonical_entrypoint_request_adapter", fake_adapter)
    globals_map = {
        "run_canonical_eval_pipeline": pipeline,
        "run_canonical_eval_entrypoint_pipeline": entrypoint_adapter,
    }
    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")

    result = eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline(
        entrypoint_globals=globals_map,
        **_canonical_wrapper_kwargs(tmp_path, parser=parser, stack=stack),
    )

    assert result == 41
    assert find_spec("weiss_rl.workflows.eval_entrypoint_support.compat") is None
    assert observed["dependency_globals"] is globals_map
    adapter_call = observed["adapter"]
    request = adapter_call["request"]
    assert request.parser is parser
    assert request.stack is stack
    assert adapter_call["run_canonical_eval_pipeline_fn"] is pipeline
    assert adapter_call["run_canonical_eval_entrypoint_pipeline_fn"] is entrypoint_adapter


def test_eval_entrypoint_runtime_canonical_wrapper_forwards_callables(monkeypatch, tmp_path: Path) -> None:
    import weiss_rl.workflows.eval_entrypoint_support.runtime as eval_entrypoint_runtime

    observed: dict[str, object] = {}
    dependencies = object()

    def fake_dependencies(_entrypoint_globals: object) -> object:
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
    globals_map = {
        "run_canonical_eval_pipeline": fake_pipeline,
        "run_canonical_eval_entrypoint_pipeline": fake_entrypoint_adapter,
    }
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_canonical_dependencies",
        fake_dependencies,
    )

    result = eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline(
        entrypoint_globals=globals_map,
        **_canonical_wrapper_kwargs(tmp_path, parser=parser, stack=stack),
    )

    assert result == 29
    assert observed["dependencies_called"] is True
    adapter_call = observed["adapter"]
    assert callable(adapter_call["canonical_dependencies_fn"])
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


def _canonical_wrapper_kwargs(tmp_path: Path, *, parser: argparse.ArgumentParser, stack: object) -> dict[str, Any]:
    return {
        "parser": parser,
        "stack": stack,
        "run_dir": tmp_path / "run",
        "final_eval_dir": tmp_path / "final",
        "policy_ids": ["B0 RandomLegal"],
        "snapshot_registry_path": tmp_path / "registry.json",
        "dev_eval_summaries_path": tmp_path / "dev.json",
        "b1_baseline_run_dir": tmp_path / "b1",
        "bootstrap_samples": 8,
        "paired_seed_limit": 1,
        "stage1_paired_seeds": 2,
        "max_paired_seeds": 3,
        "skip_metagame": True,
        "study_config_path": tmp_path / "study.yaml",
        "skip_figures": True,
        "skip_readiness": True,
        "git_commit_override": "abc123",
    }
