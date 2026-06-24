from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.artifacts.reproducibility import hash_seed_file
from weiss_rl.config import load_stack_config
from weiss_rl.eval.simulator.harness import ScheduledGame
from weiss_rl.league import run_promotion_gate

from ._config_paths import canonical_stack_config_path
from .promotion_gate_test_support import CONFIG_HASH256, RUN_ID256, SPEC_HASH256, OutcomeRunner, OutcomeToken, repo_root


def test_run_promotion_gate_writes_pass_record_in_anchor_order(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    result = run_promotion_gate(
        stack=stack,
        run_dir=tmp_path,
        focal_policy_id="candidate_snapshot",
        anchor_policy_ids={
            "B0 RandomLegal": "random_legal",
            "B1 NoLeague baseline": "b1_snapshot",
            "B2 HeuristicPublic": "heuristic_public",
        },
        runner=OutcomeRunner(lambda _: "W"),
        run_id256=RUN_ID256,
        config_hash256=CONFIG_HASH256,
        spec_hash256=SPEC_HASH256,
        sample_count=64,
        bootstrap_seed=7,
    )

    assert result.passed is True
    assert result.ordered_opponents == (
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
    )
    assert result.seed_file_sha256 == hash_seed_file(repo_root() / "configs/seeds/promotion_eval_seeds.txt")
    assert result.overall_posterior.prob_gt_target == 1.0
    assert result.truncation.rate == 0.0

    record_path = tmp_path / "promotion_gate.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["decision"] == {"passed": True, "reasons": []}
    assert payload["ordered_opponents"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
    ]
    assert payload["seed_file"] == {
        "path": "configs/seeds/promotion_eval_seeds.txt",
        "sha256": result.seed_file_sha256,
    }
    assert [anchor["episodes_path"] for anchor in payload["anchors"]] == [
        "promotion_gate_episodes/00_b0_randomlegal.jsonl",
        "promotion_gate_episodes/01_b1_noleague_baseline.jsonl",
        "promotion_gate_episodes/02_b2_heuristicpublic.jsonl",
    ]


def test_run_promotion_gate_records_machine_readable_failure_reasons(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    def outcome_for_game(game: ScheduledGame) -> OutcomeToken:
        return "L" if game.opponent_policy_id == "b1_snapshot" else "W"

    result = run_promotion_gate(
        stack=stack,
        run_dir=tmp_path,
        focal_policy_id="candidate_snapshot",
        anchor_policy_ids={
            "B0 RandomLegal": "random_legal",
            "B1 NoLeague baseline": "b1_snapshot",
        },
        runner=OutcomeRunner(outcome_for_game),
        run_id256=RUN_ID256,
        config_hash256=CONFIG_HASH256,
        spec_hash256=SPEC_HASH256,
        sample_count=128,
        bootstrap_seed=11,
    )

    assert result.passed is False
    assert {reason["code"] for reason in result.reasons} == {
        "anchor_loss_guardrail_exceeded",
        "overall_posterior_below_threshold",
    }
    assert any(
        reason["code"] == "anchor_loss_guardrail_exceeded" and reason["anchor_name"] == "B1 NoLeague baseline"
        for reason in result.reasons
    )
    assert result.truncation.rate == 0.0

    payload = json.loads((tmp_path / "promotion_gate.json").read_text(encoding="utf-8"))
    assert payload["decision"]["passed"] is False
    assert {reason["code"] for reason in payload["decision"]["reasons"]} == {
        "anchor_loss_guardrail_exceeded",
        "overall_posterior_below_threshold",
    }
