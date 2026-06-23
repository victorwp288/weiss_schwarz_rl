from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.report_publication import publish_canonical_eval_run_reports

from .canonical_eval_publication_test_support import (
    canonical_run_state,
    canonical_runtime_state,
    canonical_supplemental_outputs,
)


def test_canonical_report_publication_updates_run_level_reports(tmp_path: Path) -> None:
    layout = SimpleNamespace()
    run_state = canonical_run_state(layout=layout)
    runtime_state = canonical_runtime_state(tmp_path)
    supplemental = canonical_supplemental_outputs(tmp_path)
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
