from __future__ import annotations

from pathlib import Path

from weiss_rl.workflows.canonical_eval.cli_messages import render_canonical_eval_output_messages
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

from .canonical_eval_publication_test_support import canonical_publication_layout, canonical_runtime_state


def test_canonical_output_message_renderer_handles_optional_outputs(tmp_path: Path) -> None:
    layout = canonical_publication_layout(tmp_path)
    runtime_state = canonical_runtime_state(tmp_path)

    minimal_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(),
            readiness_payload=None,
        ),
    )

    assert minimal_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )

    full_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload={"kind": "summary"},
            figure_paths=(
                tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",
                tmp_path / "run" / "figures" / "paper" / "main_eval.png",
            ),
            readiness_payload={"passed": False},
        ),
    )

    assert full_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}",
        f"Rendered 2 paper figure files to {layout.figures_paper_dir}",
        f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}",
        "Paper readiness: failed",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )
