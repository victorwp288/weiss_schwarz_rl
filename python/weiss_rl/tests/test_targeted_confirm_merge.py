from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.targeted_confirm_merge import merge_targeted_confirm_summaries


def _write_summary(path: Path, rows: list[tuple[str, int, int]], paired_seeds: int = 256) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "paired_seeds": paired_seeds,
                "rows": [
                    {
                        "opponent_policy_id": opponent,
                        "wins": wins,
                        "games": games,
                        "mean": wins / games,
                        "summary_path": f"matchups/{opponent}/matchup_summary.json",
                    }
                    for opponent, wins, games in rows
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_merge_targeted_confirm_summaries_combines_fixed_and_learned_rows(tmp_path: Path) -> None:
    fixed = _write_summary(
        tmp_path / "fixed.json",
        [
            ("B0 RandomLegal", 512, 512),
            ("B2 HeuristicPublic", 399, 512),
        ],
    )
    learned = _write_summary(
        tmp_path / "learned.json",
        [
            ("seed_policy_000001", 328, 512),
            ("seed_policy_000005", 273, 512),
        ],
    )

    merged = merge_targeted_confirm_summaries((fixed, learned), label="selected_a015_confirm256")

    assert merged["label"] == "selected_a015_confirm256"
    assert merged["paired_seeds"] == 256
    assert len(merged["rows"]) == 4
    assert merged["anchor_subset"]["wins"] == 911
    assert merged["legacy_subset"]["wins"] == 601
    assert merged["overall"]["games"] == 2048


def test_merge_targeted_confirm_summaries_rejects_duplicate_opponents(tmp_path: Path) -> None:
    first = _write_summary(tmp_path / "first.json", [("B2 HeuristicPublic", 399, 512)])
    second = _write_summary(tmp_path / "second.json", [("B2 HeuristicPublic", 400, 512)])

    with pytest.raises(ValueError, match="duplicate opponent"):
        merge_targeted_confirm_summaries((first, second))
