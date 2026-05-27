from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.god_search_figures import write_god_search_figures, write_main_search_extra_figures


def test_write_god_search_figures_exports_tables_and_figures(tmp_path: Path) -> None:
    compare_json = tmp_path / "compare.json"
    compare_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "group": "fixed",
                        "opponent_policy_id": "B1 NoLeague baseline",
                        "baseline_wins": 6,
                        "candidate_wins": 8,
                        "delta_wins": 2,
                        "shared_games": 10,
                    },
                    {
                        "group": "learned",
                        "opponent_policy_id": "seed_c3aac2f9dc_policy_000001",
                        "baseline_wins": 5,
                        "candidate_wins": 7,
                        "delta_wins": 2,
                        "shared_games": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = write_god_search_figures(compare_json=compare_json, out_dir=tmp_path / "figures", figure_prefix="smoke")

    assert paths.row_csv.read_text(encoding="utf-8").splitlines()[0].startswith("group,opponent_id,opponent")
    summary = json.loads(paths.group_summary_json.read_text(encoding="utf-8"))
    assert summary["fixed"]["delta_wins"] == 2
    assert summary["learned"]["candidate_rate"] == 0.7
    assert "| fixed | B1 No-League | 6/10 | 8/10 | +2 | 0.800 |" in paths.row_table_md.read_text(encoding="utf-8")
    for figure_path in (
        paths.row_win_rates_png,
        paths.row_win_rates_pdf,
        paths.delta_wins_png,
        paths.delta_wins_pdf,
        paths.group_rates_png,
        paths.group_rates_pdf,
    ):
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0


def test_write_main_search_extra_figures_exports_all_supplemental_figures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    paper_dir = tmp_path / "paper"
    data_dir.mkdir()
    (data_dir / "main_search_strength_ladder.json").write_text(
        json.dumps(
            [
                {"model": "B1", "fixed": 0.68, "learned": None, "all": None},
                {"model": "Main", "fixed": 0.77, "learned": 0.56, "all": 0.64},
                {"model": "Main search", "fixed": 0.94, "learned": 0.84, "all": 0.88},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "main_search_validation_progression.json").write_text(
        json.dumps(
            [
                {"stage": "K3 confirm64", "all": 0.86, "fixed": 0.91, "learned": 0.82},
                {"stage": "K4 confirm256", "all": 0.88, "fixed": 0.94, "learned": 0.84},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "main_search_decision_changes.json").write_text(
        json.dumps({"labels": ["B1", "League policy 1"], "delta_wins": [3, 5], "changed_decisions": [3, 5]}),
        encoding="utf-8",
    )
    (data_dir / "main_search_seat_balance.json").write_text(
        json.dumps(
            [
                {"label": "B1", "delta_pp": -1.0},
                {"label": "League policy 1", "delta_pp": 2.0},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "main_search_first_second_balance.json").write_text(
        json.dumps(
            [
                {"label": "B1", "first_minus_second_pp": -3.0},
                {"label": "League policy 1", "first_minus_second_pp": 1.0},
            ]
        ),
        encoding="utf-8",
    )

    paths = write_main_search_extra_figures(data_dir=data_dir, paper_dir=paper_dir)

    for figure_path in paths.__dict__.values():
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
