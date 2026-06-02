from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.final_eval import run_canonical_final_eval_output
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import (
    CanonicalEvalSupplementalOutputs,
    build_canonical_supplemental_outputs,
)


@dataclass(frozen=True, slots=True)
class CanonicalEvalOutputBundle:
    final_eval_payload: dict[str, Any]
    supplemental: CanonicalEvalSupplementalOutputs


def build_canonical_eval_output_bundle(
    *,
    run_dir: Path,
    bootstrap_samples: int,
    skip_metagame: bool,
    skip_figures: bool,
    skip_readiness: bool,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    dependencies: Any,
) -> CanonicalEvalOutputBundle:
    final_eval_payload = run_canonical_final_eval_output(
        bootstrap_samples=bootstrap_samples,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )
    supplemental = build_canonical_supplemental_outputs(
        run_dir=run_dir,
        skip_metagame=skip_metagame,
        skip_figures=skip_figures,
        skip_readiness=skip_readiness,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )
    return CanonicalEvalOutputBundle(
        final_eval_payload=final_eval_payload,
        supplemental=supplemental,
    )


__all__ = ["CanonicalEvalOutputBundle", "build_canonical_eval_output_bundle"]
