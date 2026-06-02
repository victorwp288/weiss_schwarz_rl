from __future__ import annotations

from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs


def render_canonical_eval_output_messages(
    *,
    layout: Any,
    runtime_state: CanonicalEvalRuntimeState,
    supplemental: CanonicalEvalSupplementalOutputs,
) -> tuple[str, ...]:
    messages = [
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
    ]
    if supplemental.metagame_payload is not None:
        messages.append(f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}")
    if supplemental.figure_paths:
        messages.append(f"Rendered {len(supplemental.figure_paths)} paper figure files to {layout.figures_paper_dir}")
    if supplemental.readiness_payload is not None:
        messages.append(f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}")
        messages.append(
            "Paper readiness: " + ("passed" if bool(supplemental.readiness_payload.get("passed", False)) else "failed")
        )
    messages.append(f"Resolved policy set: {runtime_state.policy_ids}")
    return tuple(messages)


def print_canonical_eval_output_messages(
    *,
    layout: Any,
    runtime_state: CanonicalEvalRuntimeState,
    supplemental: CanonicalEvalSupplementalOutputs,
) -> None:
    for message in render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=supplemental,
    ):
        print(message)


__all__ = ["print_canonical_eval_output_messages", "render_canonical_eval_output_messages"]
