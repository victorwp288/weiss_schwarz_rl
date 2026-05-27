from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.main_league_live_probe_gate import (
    MainLeagueLiveProbeGateConfig,
    evaluate_main_league_live_probe_gate,
)


def _write_progress(path: Path, *, hard_negative_envs: float = 3.0, b2_envs: float = 2.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "league_progress_summary_v1",
                "records_seen": 1,
                "update_count_min": 1,
                "update_count_max": 1,
                "exposure_totals": {
                    "pfsp_champion_envs": 4.0,
                    "pfsp_hard_negative_envs": hard_negative_envs,
                    "pfsp_heuristic_public_envs": 2.0,
                    "pfsp_heuristic_public_variant_envs": 2.0,
                    "pfsp_noleague_baseline_envs": 2.0,
                    "pfsp_sampled_envs": 12.0,
                },
                "pool_size_max": {
                    "pfsp_champion_pool_size": 4.0,
                    "pfsp_hard_negative_pool_size": 2.0,
                    "pfsp_pool_size": 4.0,
                },
                "policy_exposure_totals": {
                    "sampled": {
                        "b2_heuristicpublic": b2_envs,
                        "b4_heuristicpubliccontrol": 2.0,
                        "seed_x_seed_b8c698d26a_seed_c3aac2f9dc_policy_000004": 1.0,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_live_probe_gate_passes_with_required_exposure(tmp_path: Path) -> None:
    progress = _write_progress(tmp_path / "progress.json")

    report = evaluate_main_league_live_probe_gate(
        MainLeagueLiveProbeGateConfig(
            league_progress_summary_json=progress,
            required_sampled_policies=(
                "B2 HeuristicPublic",
                "B4 HeuristicPublicControl",
                "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
            ),
        )
    )

    assert report["passed"] is True
    assert report["summary"]["required_sampled_policies"][2]["total_envs"] == 1.0


def test_live_probe_gate_fails_missing_hard_negative_exposure(tmp_path: Path) -> None:
    progress = _write_progress(tmp_path / "progress.json", hard_negative_envs=0.0)

    report = evaluate_main_league_live_probe_gate(MainLeagueLiveProbeGateConfig(league_progress_summary_json=progress))

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "hard_negative_exposure_below_min"


def test_live_probe_gate_fails_missing_required_sampled_policy(tmp_path: Path) -> None:
    progress = _write_progress(tmp_path / "progress.json", b2_envs=0.0)

    report = evaluate_main_league_live_probe_gate(
        MainLeagueLiveProbeGateConfig(
            league_progress_summary_json=progress,
            required_sampled_policies=("B2 HeuristicPublic",),
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "required_sampled_policy_exposure_below_min"
