from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs


def publish_canonical_eval_run_reports(
    *,
    run_dir: Path,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    final_eval_payload: dict[str, Any],
    supplemental: CanonicalEvalSupplementalOutputs,
    dependencies: Any,
) -> None:
    dependencies.update_run_level_reports_fn(
        layout=run_state.layout,
        run_dir=run_dir,
        policy_ids=runtime_state.policy_ids,
        selection_details=runtime_state.selection_details,
        final_eval_payload=final_eval_payload,
        metagame_payload=supplemental.metagame_payload,
        figure_paths=supplemental.figure_paths,
        readiness_payload=supplemental.readiness_payload,
    )


__all__ = ["publish_canonical_eval_run_reports"]
