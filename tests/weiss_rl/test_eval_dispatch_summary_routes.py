from __future__ import annotations

from types import SimpleNamespace

from .entrypoints_test_support import Path
from .eval_dispatch_test_support import dispatch_dependencies, parser, runtime_startup, seed_stack, validated_args


def test_eval_dispatch_routes_summary_only_with_dependency_bundle(tmp_path: Path, capsys) -> None:
    from weiss_rl.workflows.eval_support.dispatch.eval_dispatch import run_eval_dispatch

    observed: dict[str, object] = {}
    stack = seed_stack(tmp_path)
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_samples=17,
        bootstrap_seed=23,
    )

    def fake_summary(**kwargs: object) -> None:
        observed["summary"] = kwargs

    run_eval_dispatch(
        parser=parser(),
        args=args,
        validated=validated_args("summary"),
        startup=runtime_startup(stack),
        dependencies=dispatch_dependencies(
            run_summary_only_eval_mode_fn=fake_summary,
            load_eval_game_records_fn="load-records",
            build_matchup_export_fn="build-export",
            build_seat_advantage_diagnostics_fn="seat-diagnostics",
            write_matchup_diagnostics_json_fn="write-diagnostics",
            write_matchup_summary_csv_fn="write-csv",
            write_matchup_summary_json_fn="write-json",
        ),
    )

    call = observed["summary"]
    assert call["stack"] is stack
    assert call["episodes_jsonl"] == tmp_path / "episodes.jsonl"
    assert call["summary_json"] == tmp_path / "summary.json"
    assert call["summary_csv"] == tmp_path / "summary.csv"
    assert call["diagnostics_json"] == tmp_path / "diagnostics.json"
    assert call["bootstrap_samples"] == 17
    assert call["bootstrap_seed"] == 23
    assert call["load_eval_game_records_fn"] == "load-records"
    assert call["build_matchup_export_fn"] == "build-export"
    assert call["build_seat_advantage_diagnostics_fn"] == "seat-diagnostics"
    assert call["write_matchup_diagnostics_json_fn"] == "write-diagnostics"
    assert call["write_matchup_summary_csv_fn"] == "write-csv"
    assert call["write_matchup_summary_json_fn"] == "write-json"
    assert "Verified runtime spec bundle" in capsys.readouterr().out


def test_eval_dispatch_contract_check_only_skips_route_adapters(tmp_path: Path, capsys) -> None:
    from weiss_rl.workflows.eval_support.dispatch.eval_dispatch import run_eval_dispatch

    stack = seed_stack(tmp_path, report_eval=tmp_path / "report_eval.txt", dev_eval=tmp_path / "dev.txt")
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=None,
    )

    run_eval_dispatch(
        parser=parser(),
        args=args,
        validated=validated_args("contract_check"),
        startup=runtime_startup(stack),
        dependencies=dispatch_dependencies(),
    )

    output = capsys.readouterr().out
    assert "Verified runtime spec bundle" in output
    assert "Evaluation contract check complete; no episodes were summarized." in output
    assert "Seed sets: ['dev_eval', 'report_eval']" in output
