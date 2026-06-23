from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from weiss_rl.eval import run_final_eval

from .final_eval_test_support import (
    _CONFIG_HASH256,
    _RUN_ID256,
    _SPEC_HASH256,
    OutcomeToken,
    _FakeMatrixRunner,
    _final_eval_stop_rules,
)


def test_run_final_eval_writes_matrix_exports_and_posterior_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    policies = ["policy_beta", "policy_alpha"]
    seed_file = tmp_path / "report_eval_seeds.txt"
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    outcomes: dict[tuple[str, str, int, int], OutcomeToken] = {
        ("policy_beta", "policy_beta", 0, 0): "D",
        ("policy_beta", "policy_beta", 0, 1): "D",
        ("policy_beta", "policy_beta", 1, 0): "D",
        ("policy_beta", "policy_beta", 1, 1): "D",
        ("policy_beta", "policy_alpha", 0, 0): "W",
        ("policy_beta", "policy_alpha", 0, 1): "W",
        ("policy_beta", "policy_alpha", 1, 0): "L",
        ("policy_beta", "policy_alpha", 1, 1): "L",
        ("policy_beta", "policy_alpha", 2, 0): "W",
        ("policy_beta", "policy_alpha", 2, 1): "L",
        ("policy_alpha", "policy_alpha", 0, 0): "D",
        ("policy_alpha", "policy_alpha", 0, 1): "D",
        ("policy_alpha", "policy_alpha", 1, 0): "D",
        ("policy_alpha", "policy_alpha", 1, 1): "D",
    }
    runner = _FakeMatrixRunner(outcomes)

    payload = run_final_eval(
        output_dir=output_dir,
        runner=runner,
        policy_ids=policies,
        paired_seeds=[11, 22, 33],
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        stop_rules=_final_eval_stop_rules(),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=16,
        seed_file_path=seed_file,
        metadata={"paper_tag": "final_results_v1"},
    )

    assert payload["policy_ids"] == policies
    assert payload["metadata"]["paper_tag"] == "final_results_v1"
    assert payload["metadata"]["seed_file"]["path"] == seed_file.as_posix()
    assert payload["matrices"]["stop_reason"]["values"] == [
        ["precision", "budget"],
        ["budget", "precision"],
    ]
    assert payload["matrices"]["paired_seed_count"]["values"] == [
        [2, 3],
        [3, 2],
    ]
    assert payload["matrices"]["mean"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["mean"]["values"][0][1]
    )
    assert payload["matrices"]["ci_low"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["ci_high"]["values"][0][1]
    )
    assert payload["matrices"]["ci_high"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["ci_low"]["values"][0][1]
    )
    assert len(payload["posterior_samples"]["values"]) == 2
    assert len(payload["posterior_samples"]["values"][0][1]) == 16
    assert len(payload["posterior_samples"]["values"][1][0]) == 16
    assert all(
        forward + reverse == pytest.approx(1.0)
        for forward, reverse in zip(
            payload["posterior_samples"]["values"][0][1],
            payload["posterior_samples"]["values"][1][0],
            strict=True,
        )
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    posterior_payload = json.loads((output_dir / "posterior_samples.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    policy_set = json.loads((output_dir / "policy_set.json").read_text(encoding="utf-8"))

    assert summary["policy_ids"] == policies
    assert posterior_payload["policy_ids"] == policies
    assert metadata["selection"] == {"mode": "explicit", "policy_count": 2}
    assert metadata["matchup_artifacts"]["kind"] == "canonical_unordered_pairs_v1"
    assert policy_set == {"policy_ids": policies}
    assert len(summary["matchups"]) == 3
    assert summary["matchups"][1]["matrix_cells"] == [
        {"focal_policy_index": 0, "opponent_policy_index": 1},
        {"focal_policy_index": 1, "opponent_policy_index": 0},
    ]

    with (output_dir / "matrices" / "stop_reason.csv").open("r", encoding="utf-8") as handle:
        stop_reason_rows = list(csv.reader(handle))
    assert stop_reason_rows == [
        ["focal_policy_id", "policy_beta", "policy_alpha"],
        ["policy_beta", "precision", "budget"],
        ["policy_alpha", "budget", "precision"],
    ]

    with (output_dir / "matchups.csv").open("r", encoding="utf-8") as handle:
        matchup_rows = list(csv.DictReader(handle))
    assert len(matchup_rows) == 3
    assert matchup_rows[0]["matchup_dir"] == "matchups/00_policy_beta__vs__00_policy_beta"
    assert matchup_rows[1]["matchup_dir"] == "matchups/00_policy_beta__vs__01_policy_alpha"
    assert matchup_rows[1]["paired_seed_count"] == "3"
    assert matchup_rows[1]["observed_paired_seed_count"] == "3"
    assert matchup_rows[1]["excluded_paired_seed_count"] == "0"
    assert matchup_rows[1]["has_payoff_samples"] == "True"
    assert matchup_rows[2]["stop_reason"] == "precision"

    budget_summary = json.loads(
        (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "matchup_summary.json").read_text(
            encoding="utf-8"
        )
    )
    matchup_samples = json.loads(
        (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "posterior_samples.json").read_text(
            encoding="utf-8"
        )
    )

    assert budget_summary["stop_reason"] == "budget"
    assert budget_summary["observed_paired_seeds"] == 3
    assert budget_summary["excluded_paired_seeds"] == 0
    assert budget_summary["has_payoff_samples"] is True
    assert budget_summary["evaluation_context"]["used_paired_seeds"] == [11, 22, 33]
    assert matchup_samples["sample_count"] == 16
    assert len(matchup_samples["samples"]) == 16
    assert summary["matrices"]["observed_paired_seeds"]["values"] == [[2, 3], [3, 2]]
    assert summary["matrices"]["wins"]["values"] == [[0, 3], [3, 0]]
    assert summary["matrices"]["losses"]["values"] == [[0, 3], [3, 0]]
    assert not (output_dir / "matchups" / "01_policy_alpha__vs__00_policy_beta").exists()
    assert (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "diagnostics.json").is_file()
    assert (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "episodes.jsonl").is_file()
    assert {(call.focal_policy_id, call.opponent_policy_id) for call in runner.calls} == {
        ("policy_beta", "policy_beta"),
        ("policy_beta", "policy_alpha"),
        ("policy_alpha", "policy_alpha"),
    }
    assert len(runner.calls) == 14
