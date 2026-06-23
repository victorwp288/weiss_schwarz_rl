from __future__ import annotations

import json
from pathlib import Path

from .readiness_entrypoint_test_support import (
    run_readiness_entrypoint,
    write_ambiguous_final_eval_fixture,
    write_two_policy_final_eval_fixture,
)


def test_paper_readiness_entrypoint_writes_summary_json(tmp_path: Path) -> None:
    final_eval_dir = write_two_policy_final_eval_fixture(tmp_path)
    readiness_json = final_eval_dir / "paper_readiness_summary.json"

    result = run_readiness_entrypoint("--final-eval-dir", str(final_eval_dir), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"


def test_paper_readiness_entrypoint_requires_explicit_focal_policy_for_ambiguous_multi_policy_artifacts(
    tmp_path: Path,
) -> None:
    final_eval_dir = write_ambiguous_final_eval_fixture(tmp_path)
    readiness_json = final_eval_dir / "paper_readiness_summary.json"

    result = run_readiness_entrypoint("--final-eval-dir", str(final_eval_dir), cwd=tmp_path)

    assert result.returncode == 1
    assert "pass --focal-policy-id" in result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["checks"]["baseline_win_rate_vs_b0"]["reason"] == "ambiguous_non_baseline_focal_policy"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["eligible_non_baseline_policy_ids"] == [
        "policy_000300",
        "policy_000400",
    ]
