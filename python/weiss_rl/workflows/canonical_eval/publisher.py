from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.cli_messages import print_canonical_eval_output_messages
from weiss_rl.workflows.canonical_eval.report_publication import publish_canonical_eval_run_reports
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs
from weiss_rl.workflows.canonical_eval.tensorboard_publication import (
    begin_canonical_eval_tensorboard_logging,
    publish_canonical_eval_tensorboard_summaries,
)


def begin_canonical_eval_output_logging(*, run_state: CanonicalEvalRunState, dependencies: Any) -> None:
    begin_canonical_eval_tensorboard_logging(run_state=run_state, dependencies=dependencies)


def publish_canonical_eval_outputs(
    *,
    run_dir: Path,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    final_eval_payload: dict[str, Any],
    supplemental: CanonicalEvalSupplementalOutputs,
    dependencies: Any,
) -> None:
    layout = run_state.layout

    publish_canonical_eval_run_reports(
        run_dir=run_dir,
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload=final_eval_payload,
        supplemental=supplemental,
        dependencies=dependencies,
    )

    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=run_state.tensorboard_logger,
        final_eval_payload=final_eval_payload,
        supplemental=supplemental,
    )
    print_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=supplemental,
    )


__all__ = ["begin_canonical_eval_output_logging", "publish_canonical_eval_outputs"]
