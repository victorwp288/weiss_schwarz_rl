from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.phases import write_canonical_eval_outputs

from .canonical_eval_outputs_test_support import SyntheticCanonicalLayout, make_run_state, make_runtime_state


def test_canonical_eval_output_phase_preserves_final_eval_metadata(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    evaluation = SimpleNamespace(
        stop_rules={"minimum": 1},
        final_policy_set_selection=SimpleNamespace(folding="seat_swap_mean"),
        final_policy_set_size=2,
    )
    runtime_state = make_runtime_state(
        tmp_path=tmp_path,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        paired_seeds=[101, 202],
        paired_seed_limit=2,
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev_eval.json",
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
        run_dir=run_dir,
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=True,
        skip_readiness=True,
        run_state=make_run_state(
            run_dir=run_dir,
            layout=SyntheticCanonicalLayout(run_dir),
            evaluation=evaluation,
        ),
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
