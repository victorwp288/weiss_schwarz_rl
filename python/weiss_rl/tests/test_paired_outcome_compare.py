from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.paired_outcome_compare import (
    PairedOutcomeCompareConfig,
    compare_paired_targeted_outcomes,
)


def _write_eval_tree(root: Path, label: str, opponent: str, outcomes: list[str]) -> Path:
    matchup_dir = root / label / "matchups" / opponent.replace(" ", "_")
    matchup_dir.mkdir(parents=True)
    episodes_path = matchup_dir / "episodes.jsonl"
    records = []
    for index, outcome in enumerate(outcomes):
        records.append(
            {
                "config_hash256": "a" * 64,
                "engine_status": 0,
                "episode_index": index,
                "episode_key": f"{index:064x}",
                "episode_key64": index,
                "episode_seed": 1000 + (index // 2),
                "focal_policy_id": label,
                "focal_seat": index % 2,
                "opponent_policy_id": opponent,
                "outcome": outcome,
                "pair_index": index // 2,
                "seat0_policy_id": label if index % 2 == 0 else opponent,
                "seat1_policy_id": opponent if index % 2 == 0 else label,
                "spec_hash256": "b" * 64,
                "swap_index": index % 2,
                "terminated": True,
                "truncated": False,
                "decision_count": 100 + index,
                "pass_actions": 10 + index,
                "pass_with_nonpass_available": 3 + index,
            }
        )
    episodes_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")
    summary_path = matchup_dir / "matchup_summary.json"
    summary_path.write_text(json.dumps({"opponent_policy_id": opponent}, sort_keys=True), encoding="utf-8")
    targeted_summary = root / f"{label}_summary.json"
    targeted_summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "opponent_policy_id": opponent,
                        "wins": outcomes.count("W"),
                        "games": len(outcomes),
                        "mean": outcomes.count("W") / len(outcomes),
                        "summary_path": summary_path.as_posix(),
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return targeted_summary


def test_paired_outcome_compare_reports_flips_and_group_delta(tmp_path: Path) -> None:
    baseline = _write_eval_tree(tmp_path, "baseline", "seed_a", ["W", "W", "L", "L"])
    candidate = _write_eval_tree(tmp_path, "candidate", "seed_a", ["L", "W", "W", "L"])

    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            fixed_opponents=(),
            learned_opponents=("seed_a",),
            max_examples=10,
        )
    )

    row = report["rows"][0]
    assert row["shared_games"] == 4
    assert row["delta_wins"] == 0
    assert row["baseline_win_candidate_nonwin"] == 1
    assert row["baseline_nonwin_candidate_win"] == 1
    assert row["changed_outcome"] == 2
    assert len(row["examples"]) == 2
    assert report["groups"]["learned_opponents"]["delta_wins"] == 0


def test_paired_outcome_compare_reports_pair_index_split_buckets(tmp_path: Path) -> None:
    baseline = _write_eval_tree(tmp_path, "baseline", "seed_a", ["W", "L", "W", "L", "W", "W"])
    candidate = _write_eval_tree(tmp_path, "candidate", "seed_a", ["W", "W", "L", "L", "L", "W"])

    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            fixed_opponents=(),
            learned_opponents=("seed_a",),
            pair_index_split=2,
        )
    )

    row = report["rows"][0]
    assert row["pair_index_buckets"]["pair_index_lt_2"]["delta_wins"] == 0
    assert row["pair_index_buckets"]["pair_index_gte_2"]["delta_wins"] == -1
    learned_buckets = report["groups_by_pair_index_bucket"]["learned_opponents"]
    assert learned_buckets["pair_index_lt_2"]["shared_games"] == 4
    assert learned_buckets["pair_index_lt_2"]["delta_wins"] == 0
    assert learned_buckets["pair_index_gte_2"]["shared_games"] == 2
    assert learned_buckets["pair_index_gte_2"]["delta_wins"] == -1


def test_paired_outcome_compare_marks_missing_rows(tmp_path: Path) -> None:
    baseline = _write_eval_tree(tmp_path, "baseline", "seed_a", ["W", "L"])
    candidate = _write_eval_tree(tmp_path, "candidate", "seed_b", ["W", "L"])

    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            fixed_opponents=(),
            learned_opponents=("seed_a", "seed_b"),
        )
    )

    statuses = {row["opponent_policy_id"]: row["status"] for row in report["rows"]}
    assert statuses == {"seed_a": "missing", "seed_b": "missing"}


def test_paired_outcome_compare_matches_imported_seed_suffix_rows(tmp_path: Path) -> None:
    baseline_opponent = "seed_c3aac2f9dc_policy_000001"
    candidate_opponent = "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001"
    baseline = _write_eval_tree(tmp_path, "baseline", baseline_opponent, ["W", "L"])
    candidate = _write_eval_tree(tmp_path, "candidate", candidate_opponent, ["W", "W"])

    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            fixed_opponents=(),
            learned_opponents=(candidate_opponent,),
        )
    )

    row = report["rows"][0]
    assert row["status"] == "ok"
    assert row["opponent_policy_id"] == candidate_opponent
    assert row["baseline_opponent_policy_id"] == baseline_opponent
    assert row["candidate_opponent_policy_id"] == candidate_opponent
    assert row["delta_wins"] == 1
    assert report["groups"]["learned_opponents"]["delta_wins"] == 1


def test_paired_outcome_compare_infers_nonfixed_shared_rows_as_learned(tmp_path: Path) -> None:
    fixed = "B2 HeuristicPublic"
    learned = "seed_champion"
    baseline_fixed = _write_eval_tree(tmp_path, "baseline_fixed", fixed, ["W", "W"])
    baseline_learned = _write_eval_tree(tmp_path, "baseline_learned", learned, ["L", "L"])
    candidate_fixed = _write_eval_tree(tmp_path, "candidate_fixed", fixed, ["W", "W"])
    candidate_learned = _write_eval_tree(tmp_path, "candidate_learned", learned, ["W", "L"])
    baseline = tmp_path / "baseline_summary.json"
    candidate = tmp_path / "candidate_summary.json"
    baseline.write_text(
        json.dumps(
            {
                "rows": [
                    json.loads(baseline_fixed.read_text())["rows"][0],
                    json.loads(baseline_learned.read_text())["rows"][0],
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "rows": [
                    json.loads(candidate_fixed.read_text())["rows"][0],
                    json.loads(candidate_learned.read_text())["rows"][0],
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            fixed_opponents=(fixed,),
        )
    )

    assert report["learned_opponents"] == [learned]
    assert report["learned_opponents_inferred"] is True
    assert [row["opponent_policy_id"] for row in report["rows"]] == [fixed, learned]
    assert report["groups"]["fixed_baselines"]["delta_wins"] == 0
    assert report["groups"]["learned_opponents"]["delta_wins"] == 1
