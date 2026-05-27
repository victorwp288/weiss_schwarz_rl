from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.god_search_scorecard import GodSearchScorecardConfig, build_god_search_scorecard


def _write_compare(path: Path, *, fixed_delta: int, learned_delta: int, all_delta: int) -> None:
    payload = {
        "baseline": {"label": "base", "summary_json": "base.json"},
        "candidate": {"label": "search", "summary_json": "search.json"},
        "groups": {
            "all_compared": {"delta_wins": all_delta},
            "fixed_baselines": {"delta_wins": fixed_delta},
            "learned_opponents": {"delta_wins": learned_delta},
        },
        "rows": [
            {
                "status": "ok",
                "opponent_policy_id": "B2 HeuristicPublic",
                "delta_wins": fixed_delta,
                "baseline_wins": 3,
                "candidate_wins": 3 + fixed_delta,
                "shared_games": 8,
            },
            {
                "status": "ok",
                "opponent_policy_id": "seed_c3_policy_000004",
                "delta_wins": learned_delta,
                "baseline_wins": 4,
                "candidate_wins": 4 + learned_delta,
                "shared_games": 8,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_god_search_scorecard_allows_small_fixed_tradeoff_when_aggregate_improves(tmp_path: Path) -> None:
    compare = tmp_path / "compare.json"
    _write_compare(compare, fixed_delta=-1, learned_delta=5, all_delta=4)

    scorecard = build_god_search_scorecard(
        GodSearchScorecardConfig(
            compare_jsons=(compare,),
            fixed_opponents=("B2 HeuristicPublic",),
            min_all_delta_wins=3,
            min_fixed_delta_wins=-2,
            min_learned_delta_wins=1,
            max_fixed_row_drop_wins=2,
        )
    )

    entry = scorecard["entries"][0]
    assert entry["loose_gate"]["passed"]
    assert entry["escalation"]["decision"] == "run_confirm64"


def test_god_search_scorecard_blocks_catastrophic_fixed_drop(tmp_path: Path) -> None:
    compare = tmp_path / "compare.json"
    _write_compare(compare, fixed_delta=-3, learned_delta=8, all_delta=5)

    scorecard = build_god_search_scorecard(
        GodSearchScorecardConfig(
            compare_jsons=(compare,),
            fixed_opponents=("B2 HeuristicPublic",),
            max_fixed_row_drop_wins=2,
        )
    )

    entry = scorecard["entries"][0]
    assert not entry["loose_gate"]["passed"]
    assert entry["escalation"]["decision"] == "stop"
