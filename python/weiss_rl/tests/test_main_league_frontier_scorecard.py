from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.main_league_frontier_scorecard import (
    MAIN_LEAGUE_SENTINEL_OPPONENTS,
    MainLeagueFrontierScorecardConfig,
    build_main_league_frontier_scorecard,
)
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


def _write_compare(path: Path, rows: list[tuple[str, int, int]], shared_games: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = set(FIXED_THESIS_OPPONENTS)
    row_payloads = [
        {
            "opponent_policy_id": opponent,
            "status": "ok",
            "baseline_wins": baseline_wins,
            "candidate_wins": baseline_wins + delta_wins,
            "delta_wins": delta_wins,
            "baseline_mean": baseline_wins / shared_games,
            "candidate_mean": (baseline_wins + delta_wins) / shared_games,
            "changed_outcome": abs(delta_wins),
            "shared_games": shared_games,
        }
        for opponent, baseline_wins, delta_wins in rows
    ]
    fixed_delta = sum(delta for opponent, _, delta in rows if opponent in fixed)
    learned_delta = sum(delta for opponent, _, delta in rows if opponent not in fixed)
    path.write_text(
        json.dumps(
            {
                "baseline": {"label": "selected"},
                "candidate": {"label": "candidate"},
                "learned_opponents": [opponent for opponent, _, _ in rows if opponent not in fixed],
                "groups": {
                    "all_compared": {"delta_wins": fixed_delta + learned_delta, "changed_outcome": 0},
                    "fixed_baselines": {"delta_wins": fixed_delta, "changed_outcome": 0},
                    "learned_opponents": {"delta_wins": learned_delta, "changed_outcome": 0},
                },
                "rows": row_payloads,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_scorecard_sentinel_pass_recommends_full_confirm64(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "sentinel.json",
        [
            ("B2 HeuristicPublic", 90, 0),
            ("B4 HeuristicPublicControl", 90, 1),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 60, 0),
            ("seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected", 60, 1),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003", 60, 0),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", 60, 0),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000005", 60, 0),
        ],
    )

    scorecard = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))
    entry = scorecard["entries"][0]

    assert entry["panel_kind"] == "sentinel"
    assert entry["sentinel_gate"]["passed"] is True
    assert entry["escalation"]["decision"] == "run_full_confirm64"


def test_scorecard_sentinel_stops_on_b2_drop(tmp_path: Path) -> None:
    rows = [(opponent, 60, 0) for opponent in MAIN_LEAGUE_SENTINEL_OPPONENTS]
    rows[0] = ("B2 HeuristicPublic", 90, -1)
    report = _write_compare(tmp_path / "sentinel_drop.json", rows)

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["sentinel_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert entry["sentinel_gate"]["failures"][0]["reason"] == "sentinel_fixed_row_drop"


def test_scorecard_sentinel_stops_on_any_learned_row_drop(tmp_path: Path) -> None:
    rows = [(opponent, 60, 0) for opponent in MAIN_LEAGUE_SENTINEL_OPPONENTS]
    rows[2] = ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 60, -1)
    report = _write_compare(tmp_path / "sentinel_learned_drop.json", rows)

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["sentinel_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert entry["sentinel_gate"]["failures"][0]["reason"] == "sentinel_learned_row_drop"


def test_scorecard_padded_sentinel_uses_resolved_rows_for_panel_kind(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "sentinel_with_missing_full_rows.json",
        [
            *[(opponent, 60, 0) for opponent in MAIN_LEAGUE_SENTINEL_OPPONENTS],
            ("B0 RandomLegal", 0, 0),
            ("B1 NoLeague baseline", 0, 0),
            ("B3 HeuristicPublicAggro", 0, 0),
        ],
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        if row["opponent_policy_id"] in {"B0 RandomLegal", "B1 NoLeague baseline", "B3 HeuristicPublicAggro"}:
            row.update(
                {
                    "status": "missing",
                    "baseline_wins": None,
                    "candidate_wins": None,
                    "delta_wins": None,
                    "shared_games": None,
                    "baseline_mean": None,
                    "candidate_mean": None,
                    "changed_outcome": None,
                }
            )
    report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    entry = build_main_league_frontier_scorecard(
        MainLeagueFrontierScorecardConfig(compare_jsons=(report,), min_sentinel_learned_delta_wins=1)
    )["entries"][0]

    assert entry["panel_kind"] == "sentinel"
    assert entry["sentinel_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert entry["sentinel_gate"]["failures"][0]["reason"] == "sentinel_learned_aggregate_drop"


def test_scorecard_full_confirm64_pass_recommends_confirm128(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "full64.json",
        [
            *[(opponent, 90, 0) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 60, 1),
        ],
        shared_games=128,
    )

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["panel_kind"] == "full"
    assert entry["paired_seeds"] == 64
    assert entry["full_gate"]["passed"] is True
    assert entry["escalation"]["decision"] == "run_confirm128"


def test_scorecard_full_confirm64_stops_on_any_learned_row_drop(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "full64_learned_drop.json",
        [
            *[(opponent, 90, 0) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 60, 2),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003", 60, -1),
        ],
        shared_games=128,
    )

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["paired_seeds"] == 64
    assert entry["full_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert any(
        failure["reason"] == "full_learned_row_drop" and failure["threshold"] == 0
        for failure in entry["full_gate"]["failures"]
    )


def test_scorecard_full_confirm128_stops_on_learned_row_drop(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "full128.json",
        [
            *[(opponent, 180, 0) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 120, 3),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003", 120, -1),
        ],
        shared_games=256,
    )

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["paired_seeds"] == 128
    assert entry["full_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert any(
        failure["reason"] == "full_learned_row_drop" and failure["threshold"] == 0
        for failure in entry["full_gate"]["failures"]
    )


def test_scorecard_confirm256_gate_candidate_is_not_alias_publication(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "full256.json",
        [
            *[(opponent, 360, 0) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 300, 2),
        ],
        shared_games=512,
    )

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["paired_seeds"] == 256
    assert entry["full_gate"]["passed"] is True
    assert entry["escalation"]["decision"] == "publishable_gate_candidate"


def test_scorecard_confirm256_stops_on_any_learned_row_regression(tmp_path: Path) -> None:
    report = _write_compare(
        tmp_path / "full256_learned_regression.json",
        [
            *[(opponent, 360, 0) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000001", 300, 2),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003", 300, -1),
        ],
        shared_games=512,
    )

    entry = build_main_league_frontier_scorecard(MainLeagueFrontierScorecardConfig(compare_jsons=(report,)))["entries"][
        0
    ]

    assert entry["paired_seeds"] == 256
    assert entry["full_gate"]["passed"] is False
    assert entry["escalation"]["decision"] == "stop"
    assert any(
        failure["reason"] == "full_learned_row_drop" and failure["threshold"] == 0
        for failure in entry["full_gate"]["failures"]
    )
