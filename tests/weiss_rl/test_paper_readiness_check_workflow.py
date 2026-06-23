from __future__ import annotations

from pathlib import Path
from typing import Any

import weiss_rl.eval.paper_readiness as paper_readiness
import weiss_rl.eval.readiness.check_cli as paper_readiness_check_cli
import weiss_rl.eval.readiness.check_reporting as paper_readiness_check_reporting
import weiss_rl.eval.readiness.check_runtime as paper_readiness_check_runtime
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID


def test_paper_readiness_check_parser_defaults_output_and_thresholds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ready"
    parser = paper_readiness_check_cli.build_paper_readiness_check_parser()

    args = parser.parse_args(["--run-dir", str(run_dir), "--focal-policy-id", "  policy_000300  "])

    assert args.run_dir == run_dir
    assert args.final_eval_dir is None
    assert args.readiness_json is None
    assert paper_readiness_check_cli.default_readiness_json(
        run_dir=args.run_dir,
        final_eval_dir=args.final_eval_dir,
    ) == (run_dir / "paper_readiness_summary.json")
    assert args.focal_policy_id == "  policy_000300  "
    assert args.max_truncation_rate == paper_readiness.DEFAULT_TRUNCATION_MAX_RATE
    assert args.baseline_policy_id == RANDOM_LEGAL_POLICY_ID


def test_paper_readiness_check_runtime_threads_args_and_writes_default_json(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    parser = paper_readiness_check_cli.build_paper_readiness_check_parser()
    args = parser.parse_args(
        [
            "--final-eval-dir",
            str(final_eval_dir),
            "--focal-policy-id",
            "   ",
            "--baseline-win-rate-threshold",
            "0.61",
        ]
    )
    observed: dict[str, Any] = {}
    payload = {"passed": True, "alarms": []}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        observed["build"] = kwargs
        return payload

    def fake_write(path: Path, written_payload: dict[str, Any]) -> None:
        observed["write"] = (path, written_payload)

    result = paper_readiness_check_runtime.run_paper_readiness_check(
        args,
        build_paper_readiness_summary_fn=fake_build,
        write_paper_readiness_json_fn=fake_write,
    )

    assert result.readiness_json == final_eval_dir / "paper_readiness_summary.json"
    assert result.payload is payload
    assert observed["write"] == (final_eval_dir / "paper_readiness_summary.json", payload)
    assert observed["build"]["run_dir"] is None
    assert observed["build"]["final_eval_dir"] == final_eval_dir
    assert observed["build"]["focal_policy_id"] is None
    assert observed["build"]["baseline_win_rate_threshold"] == 0.61


def test_paper_readiness_check_reporting_formats_failure_sources() -> None:
    payload = {
        "passed": False,
        "alarms": ["baseline_win_rate_vs_b0", "manifest_contract", "final_eval_guardrails", "unknown_alarm"],
        "checks": {
            "baseline_win_rate_vs_b0": {
                "message": "pass --focal-policy-id to choose the focal policy explicitly",
            }
        },
        "manifest_contract": {"reason": "ValueError"},
        "final_eval_guardrails": {"message": "summary.json is missing"},
    }

    assert paper_readiness_check_reporting.format_failure_message(payload) == (
        "Paper readiness checks failed: "
        "baseline_win_rate_vs_b0 (pass --focal-policy-id to choose the focal policy explicitly), "
        "manifest_contract (ValueError), "
        "final_eval_guardrails (summary.json is missing), "
        "unknown_alarm"
    )
