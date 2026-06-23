from __future__ import annotations

from pathlib import Path


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
