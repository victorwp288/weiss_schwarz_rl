from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.supplemental_outputs import build_canonical_supplemental_outputs

from .canonical_eval_outputs_test_support import make_run_state, make_runtime_state


def test_canonical_supplemental_outputs_builds_thesis_artifacts_in_order(tmp_path: Path) -> None:
    calls: list[str] = []
    observed: dict[str, object] = {}
    run_dir = tmp_path / "run"
    layout = SimpleNamespace(
        final_eval_dir=run_dir / "eval" / "final_eval",
        metagame_dir=run_dir / "eval" / "metagame",
        paper_readiness_summary_path=run_dir / "paper_readiness_summary.json",
    )
    run_state = make_run_state(
        run_dir=run_dir,
        layout=layout,
        study_config=SimpleNamespace(metagame={"m": 1}, sensitivity={"s": 2}),
    )
    runtime_state = make_runtime_state(tmp_path=tmp_path, seed_file_path=tmp_path / "seeds.txt")

    def fake_metagame(**kwargs: object) -> dict[str, object]:
        calls.append("metagame")
        observed["metagame"] = kwargs
        return {"metagame": "payload"}

    def fake_figures(figure_run_dir: Path) -> tuple[Path, ...]:
        calls.append("figures")
        observed["figures"] = figure_run_dir
        return (figure_run_dir / "figures" / "paper" / "seat_bias.pdf",)

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

    outputs = build_canonical_supplemental_outputs(
        run_dir=run_dir,
        skip_metagame=False,
        skip_figures=False,
        skip_readiness=False,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(
            build_sensitivity_report_fn=fake_metagame,
            render_paper_figures_fn=fake_figures,
            ensure_run_level_report_scaffolding_fn=fake_scaffold,
            build_paper_readiness_summary_fn=fake_readiness,
            write_paper_readiness_json_fn=fake_write_readiness,
        ),
    )

    assert calls == ["metagame", "figures", "scaffold", "readiness", "write_readiness"]
    assert outputs.metagame_payload == {"metagame": "payload"}
    assert outputs.figure_paths == (run_dir / "figures" / "paper" / "seat_bias.pdf",)
    assert outputs.readiness_payload == {"passed": True}
    assert observed["metagame"] == {
        "final_eval_dir": layout.final_eval_dir,
        "out_dir": layout.metagame_dir,
        "metagame_config": {"m": 1},
        "sensitivity_config": {"s": 2},
    }
    assert observed["figures"] == run_dir
    assert observed["scaffold"] is layout
    assert observed["readiness"] == {"run_dir": run_dir, "focal_policy_id": "policy_000100"}
    assert observed["write_readiness"] == (layout.paper_readiness_summary_path, {"passed": True})
