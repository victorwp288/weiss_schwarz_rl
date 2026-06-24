from __future__ import annotations

from types import SimpleNamespace

from .entrypoints_test_support import Path
from .eval_dispatch_test_support import dispatch_dependencies, parser, runtime_startup, seed_stack, validated_args


def test_eval_dispatch_request_preserves_route_payloads(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.dispatch.eval_dispatch_request import eval_dispatch_request

    stack = seed_stack(tmp_path)
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final_eval",
        policy_id=("B0 RandomLegal", "policy_000100"),
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples="13",
        skip_metagame=1,
        study_config=tmp_path / "study.yaml",
        skip_figures=0,
        skip_readiness=True,
        git_commit_override=123,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_seed="23",
        public_demo_paired_seeds="7",
        public_demo_bootstrap_samples="11",
    )
    dependencies = dispatch_dependencies(
        public_demo_stop_rules_fn="stop-rules",
        run_public_demo_final_eval_fn="public-final",
        run_public_demo_eval_mode_fn="public-mode",
        run_canonical_eval_pipeline_fn="canonical",
        run_summary_only_eval_mode_fn="summary",
        load_eval_game_records_fn="load-records",
        build_matchup_export_fn="build-export",
        build_seat_advantage_diagnostics_fn="seat-diagnostics",
        write_matchup_diagnostics_json_fn="write-diagnostics",
        write_matchup_summary_csv_fn="write-csv",
        write_matchup_summary_json_fn="write-json",
    )

    request = eval_dispatch_request(
        parser=parser(),
        args=args,
        validated=validated_args("canonical", paired_seed_limit=5, stage1_paired_seeds=3, max_paired_seeds=9),
        startup=runtime_startup(stack),
        dependencies=dependencies,
    )

    assert request.is_public_demo is False
    assert request.has_run_dir is True
    assert request.has_episodes_jsonl is True
    assert request.public_demo_kwargs() == {
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "paired_seed_limit": 7,
        "bootstrap_samples": 11,
        "config_hash256": "c" * 64,
        "spec_hash256": "e" * 64,
        "public_demo_stop_rules_fn": "stop-rules",
        "run_public_demo_final_eval_fn": "public-final",
    }
    assert request.canonical_kwargs() == {
        "parser": request.parser,
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "snapshot_registry_path": (tmp_path / "registry.json").resolve(),
        "dev_eval_summaries_path": (tmp_path / "dev_eval.json").resolve(),
        "b1_baseline_run_dir": (tmp_path / "b1").resolve(),
        "bootstrap_samples": 13,
        "paired_seed_limit": 5,
        "stage1_paired_seeds": 3,
        "max_paired_seeds": 9,
        "skip_metagame": True,
        "study_config_path": (tmp_path / "study.yaml").resolve(),
        "skip_figures": False,
        "skip_readiness": True,
        "git_commit_override": "123",
    }
    assert request.summary_only_kwargs() == {
        "stack": stack,
        "episodes_jsonl": tmp_path / "episodes.jsonl",
        "summary_json": tmp_path / "summary.json",
        "summary_csv": tmp_path / "summary.csv",
        "diagnostics_json": tmp_path / "diagnostics.json",
        "bootstrap_samples": 13,
        "bootstrap_seed": 23,
        "load_eval_game_records_fn": "load-records",
        "build_matchup_export_fn": "build-export",
        "build_seat_advantage_diagnostics_fn": "seat-diagnostics",
        "write_matchup_diagnostics_json_fn": "write-diagnostics",
        "write_matchup_summary_csv_fn": "write-csv",
        "write_matchup_summary_json_fn": "write-json",
    }
