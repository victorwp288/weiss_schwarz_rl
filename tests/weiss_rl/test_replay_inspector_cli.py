from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.replay import inspector_entrypoint

from ._config_paths import canonical_stack_config_path


def test_replay_inspector_cli_main_supports_json_stdout_and_report_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    canned_report = {
        "bundle_path": "bundle.zip",
        "policy_a": {"label": "A", "weights_path": "a.pt"},
        "policy_b": {"label": "B", "weights_path": "b.pt"},
        "summary": {
            "compared_steps": 1,
            "top_k": 1,
            "max_total_variation": 0.5,
            "mean_total_variation": 0.5,
            "median_total_variation": 0.5,
            "max_abs_probability_delta": 0.5,
        },
        "top_differences": [],
        "compared_steps": 1,
    }
    captured_kwargs: dict[str, object] = {}

    def fake_inspect_replay_bundle(**_: object) -> dict[str, object]:
        captured_kwargs.update(_)
        return canned_report

    monkeypatch.setattr(inspector_entrypoint, "inspect_replay_bundle", fake_inspect_replay_bundle)

    exit_code = inspector_entrypoint.main(
        [
            "--bundle",
            str(tmp_path / "bundle.zip"),
            "--stack-config",
            str(canonical_stack_config_path()),
            "--policy-a",
            "policy_a",
            "--policy-b",
            "policy_b",
            "--json",
            "--report-json",
            str(report_path),
            "--opponent-context-policy-id",
            "B2 HeuristicPublic",
            "--require-opponent-context-index",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["opponent_context_policy_id"] == "B2 HeuristicPublic"
    assert captured_kwargs["require_opponent_context_index"] is True
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == canned_report
    assert json.loads(report_path.read_text(encoding="utf-8")) == canned_report
