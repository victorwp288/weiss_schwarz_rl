from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import weiss_rl.workflows.canonical_eval.output_bundle as output_bundle_module
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

from .canonical_eval_outputs_test_support import make_run_state, make_runtime_state


def test_canonical_output_bundle_builds_final_eval_before_supplemental_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    run_dir = tmp_path / "run"
    run_state = make_run_state(run_dir=run_dir)
    runtime_state = make_runtime_state(tmp_path=tmp_path)

    def fake_final_eval(**kwargs: object) -> dict[str, object]:
        calls.append("final")
        assert kwargs["bootstrap_samples"] == 8
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return {"final": "payload"}

    def fake_supplemental(**kwargs: object) -> CanonicalEvalSupplementalOutputs:
        calls.append("supplemental")
        assert kwargs["run_dir"] == run_dir
        assert kwargs["skip_metagame"] is True
        assert kwargs["skip_figures"] is False
        assert kwargs["skip_readiness"] is True
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(run_dir / "figures" / "paper" / "seat_bias.pdf",),
            readiness_payload=None,
        )

    monkeypatch.setattr(output_bundle_module, "run_canonical_final_eval_output", fake_final_eval)
    monkeypatch.setattr(output_bundle_module, "build_canonical_supplemental_outputs", fake_supplemental)

    bundle = output_bundle_module.build_canonical_eval_output_bundle(
        run_dir=run_dir,
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=False,
        skip_readiness=True,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(),
    )

    assert calls == ["final", "supplemental"]
    assert bundle.final_eval_payload == {"final": "payload"}
    assert bundle.supplemental.figure_paths == (run_dir / "figures" / "paper" / "seat_bias.pdf",)
