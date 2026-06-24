from __future__ import annotations

from types import SimpleNamespace

from .entrypoints_test_support import Path, pytest
from .eval_dispatch_test_support import dispatch_dependencies, parser, runtime_startup, seed_stack, validated_args


def test_eval_dispatch_routes_canonical_with_normalized_args(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.dispatch.eval_dispatch import run_eval_dispatch

    observed: dict[str, object] = {}
    stack = seed_stack(tmp_path)
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        policy_id=["B0 RandomLegal", "policy_000100"],
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=13,
        skip_metagame=True,
        study_config=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        episodes_jsonl=None,
    )

    def fake_canonical(**kwargs: object) -> int:
        observed["canonical"] = kwargs
        return 23

    with pytest.raises(SystemExit) as exc_info:
        run_eval_dispatch(
            parser=parser(),
            args=args,
            validated=validated_args("canonical", paired_seed_limit=5, stage1_paired_seeds=3, max_paired_seeds=9),
            startup=runtime_startup(stack),
            dependencies=dispatch_dependencies(run_canonical_eval_pipeline_fn=fake_canonical),
        )

    assert exc_info.value.code == 23
    call = observed["canonical"]
    assert call["stack"] is stack
    assert call["run_dir"] == (tmp_path / "run").resolve()
    assert call["final_eval_dir"] == (tmp_path / "run" / "eval" / "final_eval").resolve()
    assert call["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert call["snapshot_registry_path"] == (tmp_path / "registry.json").resolve()
    assert call["dev_eval_summaries_path"] == (tmp_path / "dev_eval.json").resolve()
    assert call["b1_baseline_run_dir"] == (tmp_path / "b1").resolve()
    assert call["bootstrap_samples"] == 13
    assert call["paired_seed_limit"] == 5
    assert call["stage1_paired_seeds"] == 3
    assert call["max_paired_seeds"] == 9
    assert call["skip_metagame"] is True
    assert call["study_config_path"] == (tmp_path / "study.yaml").resolve()
    assert call["skip_figures"] is True
    assert call["skip_readiness"] is True
    assert call["git_commit_override"] == "abc123"
