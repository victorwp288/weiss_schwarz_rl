from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.output_bundle import build_canonical_eval_output_bundle
from weiss_rl.workflows.canonical_eval.publisher import (
    begin_canonical_eval_output_logging,
    publish_canonical_eval_outputs,
)
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState


def write_canonical_eval_outputs(
    *,
    run_dir: Path,
    bootstrap_samples: int,
    skip_metagame: bool,
    skip_figures: bool,
    skip_readiness: bool,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    dependencies: Any,
) -> int:
    begin_canonical_eval_output_logging(run_state=run_state, dependencies=dependencies)
    output_bundle = build_canonical_eval_output_bundle(
        run_dir=run_dir,
        bootstrap_samples=bootstrap_samples,
        skip_metagame=skip_metagame,
        skip_figures=skip_figures,
        skip_readiness=skip_readiness,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )

    publish_canonical_eval_outputs(
        run_dir=run_dir,
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload=output_bundle.final_eval_payload,
        supplemental=output_bundle.supplemental,
        dependencies=dependencies,
    )
    return 0


__all__ = ["write_canonical_eval_outputs"]
