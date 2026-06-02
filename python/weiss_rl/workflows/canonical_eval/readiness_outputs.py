from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState


def build_canonical_readiness_output(
    *,
    run_dir: Path,
    layout: Any,
    runtime_state: CanonicalEvalRuntimeState,
    dependencies: Any,
) -> dict[str, Any]:
    readiness_payload = dependencies.build_paper_readiness_summary_fn(
        run_dir=run_dir,
        focal_policy_id=runtime_state.recommended_focal_policy_id,
    )
    dependencies.write_paper_readiness_json_fn(layout.paper_readiness_summary_path, readiness_payload)
    return readiness_payload


__all__ = ["build_canonical_readiness_output"]
