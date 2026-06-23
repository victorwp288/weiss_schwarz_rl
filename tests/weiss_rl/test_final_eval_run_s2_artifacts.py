from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.eval import run_final_eval

from .final_eval_test_support import (
    _CONFIG_HASH256,
    _RUN_ID256,
    _SPEC_HASH256,
    OutcomeToken,
    _FakeMatrixRunner,
    _final_eval_stop_rules,
)


def test_run_final_eval_emits_explicit_s2_no_included_pair_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    policies = ["policy_alpha", "policy_beta"]
    outcomes: dict[tuple[str, str, int, int], OutcomeToken] = {
        (focal_policy_id, opponent_policy_id, pair_index, swap_index): "T"
        for focal_policy_id in policies
        for opponent_policy_id in policies
        for pair_index in range(2)
        for swap_index in (0, 1)
    }
    runner = _FakeMatrixRunner(outcomes)

    payload = run_final_eval(
        output_dir=output_dir,
        runner=runner,
        policy_ids=policies,
        paired_seeds=[11, 22],
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        stop_rules=_final_eval_stop_rules(),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S2",
        sample_count=16,
    )

    assert payload["matrices"]["paired_seed_count"]["values"] == [[0, 0], [0, 0]]
    assert payload["matrices"]["observed_paired_seeds"]["values"] == [[2, 2], [2, 2]]
    assert payload["matrices"]["excluded_paired_seeds"]["values"] == [[2, 2], [2, 2]]
    assert payload["matrices"]["has_payoff_samples"]["values"] == [[False, False], [False, False]]
    assert payload["matrices"]["stop_reason"]["values"] == [
        ["no_included_pairs", "no_included_pairs"],
        ["no_included_pairs", "no_included_pairs"],
    ]
    assert payload["posterior_samples"]["values"] == [[[], []], [[], []]]

    matchup_summary = json.loads(
        (output_dir / "matchups" / "00_policy_alpha__vs__01_policy_beta" / "matchup_summary.json").read_text(
            encoding="utf-8"
        )
    )
    matchup_samples = json.loads(
        (output_dir / "matchups" / "00_policy_alpha__vs__01_policy_beta" / "posterior_samples.json").read_text(
            encoding="utf-8"
        )
    )

    assert matchup_summary["summary"]["truncations"] == 4
    assert matchup_summary["paired_seeds"] == 0
    assert matchup_summary["observed_paired_seeds"] == 2
    assert matchup_summary["excluded_paired_seeds"] == 2
    assert matchup_summary["has_payoff_samples"] is False
    assert matchup_summary["stop_reason"] == "no_included_pairs"
    assert matchup_summary["uncertainty"] == {
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "ci_half_width": None,
        "prob_gt_half": None,
        "prob_lt_half": None,
        "paired_seed_count": 0,
        "sample_count": 0,
    }
    assert matchup_samples["requested_sample_count"] == 16
    assert matchup_samples["sample_count"] == 0
    assert matchup_samples["has_payoff_samples"] is False
    assert matchup_samples["samples"] == []
