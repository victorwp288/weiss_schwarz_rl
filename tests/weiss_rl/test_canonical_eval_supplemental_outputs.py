from __future__ import annotations

from pathlib import Path


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
