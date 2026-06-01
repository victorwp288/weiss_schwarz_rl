from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from weiss_rl.experiments.paired_outcome_compare_gate import (
    PairedOutcomeCompareGateConfig,
    evaluate_paired_outcome_compare_gate,
)
from weiss_rl.experiments.paired_outcome_compare_gate_reporting import (
    paired_outcome_compare_gate_output_payload,
)
from weiss_rl.experiments.paired_outcome_compare_gate_runtime import (
    PairedOutcomeCompareGateRunResult,
    paired_outcome_compare_gate_config_from_args,
)


def test_paired_outcome_compare_gate_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_compare_gate,
        paired_outcome_compare_gate_cli,
        paired_outcome_compare_gate_entrypoint,
        paired_outcome_compare_gate_runtime,
    )

    assert paired_outcome_compare_gate_entrypoint._build_parser is (
        paired_outcome_compare_gate_cli.build_paired_outcome_compare_gate_parser
    )
    assert paired_outcome_compare_gate_entrypoint.run_paired_outcome_compare_gate is (
        paired_outcome_compare_gate_runtime.run_paired_outcome_compare_gate
    )
    assert paired_outcome_compare_gate_entrypoint.PairedOutcomeCompareGateConfig is (
        paired_outcome_compare_gate.PairedOutcomeCompareGateConfig
    )
    assert paired_outcome_compare_gate_entrypoint.evaluate_paired_outcome_compare_gate is (
        paired_outcome_compare_gate.evaluate_paired_outcome_compare_gate
    )
    assert paired_outcome_compare_gate_entrypoint.write_paired_outcome_compare_gate is (
        paired_outcome_compare_gate.write_paired_outcome_compare_gate
    )


def test_paired_outcome_compare_gate_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_compare_gate_cli import (
        build_paired_outcome_compare_gate_parser,
    )

    args = build_paired_outcome_compare_gate_parser().parse_args(
        [
            "--compare-json",
            str(tmp_path / "compare.json"),
            "--output-json",
            str(tmp_path / "gate.json"),
        ]
    )

    assert args.compare_json == [tmp_path / "compare.json"]
    assert args.output_json == tmp_path / "gate.json"
    assert args.min_all_delta_wins == 0
    assert args.min_fixed_delta_wins == 0
    assert args.min_learned_delta_wins == 0
    assert args.max_fixed_row_drop_wins == 0
    assert args.max_learned_row_drop_wins == 0
    assert args.required_opponent == []


def test_paired_outcome_compare_gate_runtime_maps_args_and_exit_codes(tmp_path: Path) -> None:
    args = SimpleNamespace(
        compare_json=[tmp_path / "a.json", tmp_path / "b.json"],
        min_all_delta_wins=1,
        min_fixed_delta_wins=2,
        min_learned_delta_wins=3,
        max_fixed_row_drop_wins=4,
        max_learned_row_drop_wins=5,
        required_opponent=["B2 HeuristicPublic", "policy_000004"],
    )

    config = paired_outcome_compare_gate_config_from_args(args)

    assert config == PairedOutcomeCompareGateConfig(
        compare_jsons=(tmp_path / "a.json", tmp_path / "b.json"),
        min_all_delta_wins=1,
        min_fixed_delta_wins=2,
        min_learned_delta_wins=3,
        max_fixed_row_drop_wins=4,
        max_learned_row_drop_wins=5,
        required_opponents=("B2 HeuristicPublic", "policy_000004"),
    )
    assert PairedOutcomeCompareGateRunResult(output_json=tmp_path / "gate.json", report={"passed": True}).exit_code == 0
    assert (
        PairedOutcomeCompareGateRunResult(output_json=tmp_path / "gate.json", report={"passed": False}).exit_code == 2
    )


def test_paired_outcome_compare_gate_reporting_preserves_compact_console_payload(tmp_path: Path) -> None:
    report = {
        "passed": False,
        "failures": [{"reason": "fixed_aggregate_drop"}],
        "summary": {"entry_count": 1, "passed_count": 0, "failed_count": 1},
    }

    assert paired_outcome_compare_gate_output_payload(output_json=tmp_path / "gate.json", report=report) == {
        "output_json": (tmp_path / "gate.json").as_posix(),
        "passed": False,
        "failures": [{"reason": "fixed_aggregate_drop"}],
        "summary": {"entry_count": 1, "passed_count": 0, "failed_count": 1},
    }


def test_paired_outcome_compare_gate_passes_no_regression_screen(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=0,
        learned_delta=1,
        rows=[
            ("B2 HeuristicPublic", "fixed", 0),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", "learned", 1),
        ],
    )

    report = evaluate_paired_outcome_compare_gate(
        PairedOutcomeCompareGateConfig(
            compare_jsons=(compare,),
            required_opponents=("B2 HeuristicPublic", "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004"),
        )
    )

    assert report["passed"] is True
    assert report["entries"][0]["group_deltas"]["learned_delta_wins"] == 1


def test_paired_outcome_compare_gate_fails_fixed_and_learned_row_drops(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=-3,
        learned_delta=-1,
        rows=[
            ("B2 HeuristicPublic", "fixed", -3),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", "learned", -1),
        ],
    )

    report = evaluate_paired_outcome_compare_gate(PairedOutcomeCompareGateConfig(compare_jsons=(compare,)))

    reasons = [failure["reason"] for failure in report["failures"]]
    assert report["passed"] is False
    assert "fixed_aggregate_drop" in reasons
    assert "learned_aggregate_drop" in reasons
    assert "fixed_row_drop" in reasons
    assert "learned_row_drop" in reasons


def test_paired_outcome_compare_gate_requires_expected_opponents(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=0,
        learned_delta=0,
        rows=[("B2 HeuristicPublic", "fixed", 0)],
    )

    report = evaluate_paired_outcome_compare_gate(
        PairedOutcomeCompareGateConfig(
            compare_jsons=(compare,),
            required_opponents=("B2 HeuristicPublic", "policy_000004"),
        )
    )

    assert report["passed"] is False
    assert any(failure["reason"] == "missing_required_opponent" for failure in report["failures"])


def _write_compare(
    path: Path,
    *,
    fixed_delta: int,
    learned_delta: int,
    rows: list[tuple[str, str, int]],
) -> Path:
    fixed_opponents = [opponent for opponent, group, _delta in rows if group == "fixed"]
    learned_opponents = [opponent for opponent, group, _delta in rows if group == "learned"]
    payload_rows = [
        {
            "baseline_opponent_policy_id": opponent,
            "candidate_opponent_policy_id": opponent,
            "delta_wins": delta,
            "baseline_wins": 8,
            "candidate_wins": 8 + delta,
            "shared_games": 16,
        }
        for opponent, _group, delta in rows
    ]
    path.write_text(
        json.dumps(
            {
                "kind": "paired_targeted_outcome_compare_v1",
                "baseline": {"label": "selected"},
                "candidate": {"label": "candidate"},
                "fixed_opponents": fixed_opponents,
                "learned_opponents": learned_opponents,
                "groups": {
                    "all_compared": {"delta_wins": fixed_delta + learned_delta},
                    "fixed_baselines": {"delta_wins": fixed_delta},
                    "learned_opponents": {"delta_wins": learned_delta},
                },
                "rows": payload_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
