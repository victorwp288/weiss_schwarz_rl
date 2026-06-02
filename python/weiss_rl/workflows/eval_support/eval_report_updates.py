from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.workflows.eval_support.eval_report_io import _load_json_object, _write_json
from weiss_rl.workflows.eval_support.eval_report_scaffolding import (
    _load_determinism_report_or_default,
    _load_run_summary_or_default,
)
from weiss_rl.workflows.eval_support.eval_report_update_payloads import (
    RunLevelReportUpdateInputs,
    build_determinism_report_update_fields,
    build_run_summary_update_fields,
)


def _update_run_level_reports(
    *,
    layout: ArtifactLayout,
    run_dir: Path,
    policy_ids: list[str],
    selection_details: dict[str, Any],
    final_eval_payload: dict[str, Any],
    metagame_payload: dict[str, Any] | None,
    figure_paths: tuple[Path, ...],
    readiness_payload: dict[str, Any] | None,
) -> None:
    inputs = RunLevelReportUpdateInputs(
        layout=layout,
        run_dir=run_dir,
        policy_ids=policy_ids,
        selection_details=selection_details,
        final_eval_payload=final_eval_payload,
        metagame_payload=metagame_payload,
        figure_paths=figure_paths,
        readiness_payload=readiness_payload,
    )

    run_summary = _load_run_summary_or_default(layout)
    run_summary.update(build_run_summary_update_fields(inputs))
    _write_json(layout.run_summary_path, run_summary)

    determinism_report = _load_determinism_report_or_default(layout)
    replay_verification = _load_json_object(layout.replay_verification_json(), label="replay verification summary")
    artifact_hashes = _load_json_object(layout.final_eval_aggregate_hashes_json(), label="final eval artifact hashes")
    determinism_report.update(
        build_determinism_report_update_fields(
            inputs,
            replay_verification=replay_verification,
            artifact_hashes=artifact_hashes,
        )
    )
    _write_json(layout.determinism_report_path, determinism_report)


__all__ = ["_update_run_level_reports"]
