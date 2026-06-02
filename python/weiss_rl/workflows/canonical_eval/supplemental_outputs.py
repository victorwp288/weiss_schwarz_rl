from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.figure_outputs import build_canonical_figure_outputs
from weiss_rl.workflows.canonical_eval.metagame_outputs import build_canonical_metagame_output
from weiss_rl.workflows.canonical_eval.readiness_outputs import build_canonical_readiness_output
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState


@dataclass(frozen=True, slots=True)
class CanonicalEvalSupplementalOutputs:
    metagame_payload: dict[str, Any] | None
    figure_paths: tuple[Path, ...]
    readiness_payload: dict[str, Any] | None


def build_canonical_supplemental_outputs(
    *,
    run_dir: Path,
    skip_metagame: bool,
    skip_figures: bool,
    skip_readiness: bool,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    dependencies: Any,
) -> CanonicalEvalSupplementalOutputs:
    layout = run_state.layout

    metagame_payload: dict[str, Any] | None = None
    if not skip_metagame:
        assert run_state.study_config is not None
        metagame_payload = build_canonical_metagame_output(
            layout=layout,
            study_config=run_state.study_config,
            dependencies=dependencies,
        )

    figure_paths: tuple[Path, ...] = ()
    if not skip_figures:
        figure_paths = build_canonical_figure_outputs(run_dir=run_dir, dependencies=dependencies)

    dependencies.ensure_run_level_report_scaffolding_fn(layout)

    readiness_payload: dict[str, Any] | None = None
    if not skip_readiness:
        readiness_payload = build_canonical_readiness_output(
            run_dir=run_dir,
            layout=layout,
            runtime_state=runtime_state,
            dependencies=dependencies,
        )

    return CanonicalEvalSupplementalOutputs(
        metagame_payload=metagame_payload,
        figure_paths=figure_paths,
        readiness_payload=readiness_payload,
    )


__all__ = ["CanonicalEvalSupplementalOutputs", "build_canonical_supplemental_outputs"]
