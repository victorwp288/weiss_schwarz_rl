from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.publisher import (
    begin_canonical_eval_output_logging,
    publish_canonical_eval_outputs,
)

from .canonical_eval_publication_test_support import (
    FakeTensorBoardLogger,
    canonical_publication_layout,
    canonical_run_state,
    canonical_runtime_state,
    canonical_supplemental_outputs,
)


def test_canonical_output_publisher_updates_reports_tensorboard_and_cli(tmp_path: Path, capsys) -> None:
    tensorboard_logger = FakeTensorBoardLogger()
    layout = canonical_publication_layout(tmp_path)
    run_state = canonical_run_state(layout=layout, tensorboard_logger=tensorboard_logger)
    runtime_state = canonical_runtime_state(tmp_path)
    supplemental = canonical_supplemental_outputs(tmp_path)
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
