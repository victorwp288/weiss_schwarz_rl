from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.targeted_confirm_prefix import (
    TargetedConfirmPrefixConfig,
    derive_targeted_confirm_prefix_summary,
)


def test_derive_targeted_confirm_prefix_summary_uses_prefix_pairs(tmp_path: Path) -> None:
    summary_json = _write_targeted_fixture(
        tmp_path,
        opponent="B2 HeuristicPublic",
        outcomes=("W", "L", "W", "W", "L", "W"),
    )
    output_json = tmp_path / "eval" / "confirm2" / "targeted_confirm2_summary.json"

    summary = derive_targeted_confirm_prefix_summary(
        TargetedConfirmPrefixConfig(
            source_summary_json=summary_json,
            output_summary_json=output_json,
            paired_seeds=2,
        )
    )

    assert summary["paired_seeds"] == 2
    assert summary["games_per_row"] == 4
    assert summary["rows"][0]["wins"] == 3
    assert summary["rows"][0]["games"] == 4
    assert summary["rows"][0]["mean"] == 0.75
    assert summary["overall"] == {"games": 4, "mean": 0.75, "wins": 3}
    assert json.loads(output_json.read_text(encoding="utf-8"))["derived_from"]["paired_seed_prefix"] == 2


def test_derive_targeted_confirm_prefix_summary_requires_complete_pairs(tmp_path: Path) -> None:
    summary_json = _write_targeted_fixture(
        tmp_path,
        opponent="B2 HeuristicPublic",
        outcomes=("W", "L"),
    )

    with pytest.raises(ValueError, match="expected 2 paired seeds"):
        derive_targeted_confirm_prefix_summary(
            TargetedConfirmPrefixConfig(
                source_summary_json=summary_json,
                output_summary_json=tmp_path / "out.json",
                paired_seeds=2,
            )
        )


def _write_targeted_fixture(tmp_path: Path, *, opponent: str, outcomes: tuple[str, ...]) -> Path:
    matchup_dir = tmp_path / "eval" / "confirm3" / "matchups" / "00_focal__vs__01_opponent"
    matchup_dir.mkdir(parents=True)
    episodes_path = matchup_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for index, outcome in enumerate(outcomes):
            pair_index = index // 2
            swap_index = index % 2
            handle.write(
                json.dumps(
                    {
                        "config_hash256": "a" * 64,
                        "decision_count": 1,
                        "engine_status": 0,
                        "episode_index": index,
                        "episode_key": f"key-{index}",
                        "episode_key64": index + 1,
                        "episode_seed": 100 + pair_index,
                        "focal_policy_id": "focal",
                        "focal_seat": swap_index,
                        "main_move_actions": 0,
                        "max_consecutive_main_moves": 0,
                        "no_progress_count": 0,
                        "opponent_policy_id": opponent,
                        "outcome": outcome,
                        "pair_index": pair_index,
                        "pass_actions": 0,
                        "pass_with_nonpass_available": 0,
                        "run_id256": "b" * 64,
                        "seat0_deck": "preset:main_deck_5hy_yotsuba_v1",
                        "seat0_policy_id": "focal" if swap_index == 0 else opponent,
                        "seat1_deck": "preset:main_deck_5hy_yotsuba_v1",
                        "seat1_policy_id": opponent if swap_index == 0 else "focal",
                        "spec_hash256": "c" * 64,
                        "swap_index": swap_index,
                        "terminated": True,
                        "termination_reason": "terminated",
                        "tick_count": 1,
                        "total_actions": 1,
                        "truncated": False,
                    }
                )
                + "\n"
            )
    summary_json = tmp_path / "eval" / "confirm3" / "targeted_confirm3_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "focal_policy_id": "focal",
                "paired_seeds": 3,
                "games_per_row": 6,
                "rows": [
                    {
                        "focal_policy_id": "focal",
                        "opponent_policy_id": opponent,
                        "paired_seeds": 3,
                        "games": len(outcomes),
                        "wins": sum(1 for outcome in outcomes if outcome == "W"),
                        "losses": sum(1 for outcome in outcomes if outcome == "L"),
                        "draws": 0,
                        "truncations": 0,
                        "engine_errors": 0,
                        "mean": sum(1 for outcome in outcomes if outcome == "W") / len(outcomes),
                        "ci_low": 0.0,
                        "ci_high": 1.0,
                        "prob_gt_half": 0.5,
                        "summary_path": (matchup_dir / "matchup_summary.json").as_posix(),
                        "diagnostics_path": (matchup_dir / "diagnostics.json").as_posix(),
                    }
                ],
                "overall": {"wins": 0, "games": 0},
                "anchor_subset": {"wins": 0, "games": 0},
                "legacy_subset": {"wins": 0, "games": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (matchup_dir / "matchup_summary.json").write_text("{}\n", encoding="utf-8")
    return summary_json
