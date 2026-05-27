from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.b1_candidate_selection import (
    build_b1_candidate_selection,
    load_b1_dev_eval_records,
    load_reference_anchor_scores,
    publish_b1_baseline_alias,
    publish_selected_candidate_alias,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_registry(run_dir: Path, updates: tuple[int, ...]) -> None:
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": f"policy_{index:06d}",
                    "update": update,
                    "path": f"training/snapshots/policy_{index:06d}/weights.pt",
                    "weights_sha256": f"sha-{index}",
                }
                for index, update in enumerate(updates, start=1)
            ],
        },
    )


def test_b1_candidate_selection_maps_train_policy_to_snapshot_and_flags_falloff(tmp_path: Path) -> None:
    good = tmp_path / "good_seed"
    weak = tmp_path / "weak_seed"
    _write_registry(good, (25, 50, 75))
    _write_registry(weak, (25, 50))
    _write_json(
        good / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {
                "update_count": 25,
                "policy_version": 1,
                "aggregate_score": 0.55,
                "anchor_scores": {
                    "B0 RandomLegal": 1.0,
                    "B2 HeuristicPublic": 0.45,
                    "B3 HeuristicPublicAggro": 0.43,
                    "B4 HeuristicPublicControl": 0.40,
                },
            },
            "train_u50_p2": {
                "update_count": 50,
                "policy_version": 2,
                "aggregate_score": 0.67,
                "anchor_scores": {
                    "B0 RandomLegal": 1.0,
                    "B2 HeuristicPublic": 0.65,
                    "B3 HeuristicPublicAggro": 0.53,
                    "B4 HeuristicPublicControl": 0.50,
                },
            },
            "train_u75_p3": {
                "update_count": 75,
                "policy_version": 3,
                "aggregate_score": 0.60,
                "anchor_scores": {
                    "B0 RandomLegal": 1.0,
                    "B2 HeuristicPublic": 0.55,
                    "B3 HeuristicPublicAggro": 0.45,
                    "B4 HeuristicPublicControl": 0.50,
                },
            },
        },
    )
    _write_json(
        weak / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u50_p2": {
                "update_count": 50,
                "policy_version": 2,
                "aggregate_score": 0.54,
                "anchor_scores": {
                    "B0 RandomLegal": 1.0,
                    "B2 HeuristicPublic": 0.44,
                    "B3 HeuristicPublicAggro": 0.31,
                    "B4 HeuristicPublicControl": 0.41,
                },
            }
        },
    )
    _write_json(
        good / "eval" / "dev_eval_confirmatory" / "update_50" / "summary.json",
        {
            "aggregate_score": 0.71,
            "anchor_scores": {
                "B2 HeuristicPublic": 0.65,
                "B3 HeuristicPublicAggro": 0.64,
                "B4 HeuristicPublicControl": 0.58,
            },
        },
    )

    summary = build_b1_candidate_selection(
        [weak, good],
        stack_config=tmp_path / "configs" / "b1.yaml",
        falloff_warning_threshold=0.05,
    )

    selected = summary["selected"]
    assert selected["run_name"] == "good_seed"
    assert selected["train_policy_id"] == "train_u50_p2"
    assert selected["snapshot_policy_id"] == "policy_000002"
    assert selected["snapshot_path"] == "training/snapshots/policy_000002/weights.pt"
    assert selected["eligible"] is True
    assert selected["selection_score_source"] == "confirmatory_dev_eval"
    assert selected["dev_eval_required_anchor_min"] == 0.50
    assert selected["required_anchor_min"] == 0.58
    assert selected["confirmatory_dev_eval"]["aggregate_score"] == 0.71
    assert selected["confirmatory_dev_eval"]["anchor_scores"]["B4 HeuristicPublicControl"] == 0.58
    assert any("good_seed fell off" in warning for warning in summary["warnings"])
    output_flag_index = selected["confirmation_command"].index("--output-subdir")
    assert selected["confirmation_command"][output_flag_index + 1] == "b1_candidate_confirm64_policy_000002"

    source_weights = good / "training" / "snapshots" / "policy_000002" / "weights.pt"
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    source_weights.write_bytes(b"weights")
    _write_json(good / "manifest.json", {"config_canonical": {"config": {"experiment": {"role": "baseline_noleague"}}}})
    published = publish_b1_baseline_alias(
        run_dir=good,
        source_policy_id=str(selected["snapshot_policy_id"]),
        selection_summary={"selected": selected["snapshot_policy_id"]},
    )
    registry = json.loads((good / "training" / "snapshots" / "registry.json").read_text(encoding="utf-8"))
    snapshots_by_id = {snapshot["policy_id"]: snapshot for snapshot in registry["snapshots"]}
    assert published["policy_id"] == "b1_noleague_baseline"
    assert snapshots_by_id["b1_noleague_baseline"]["path"] == ("training/snapshots/b1_noleague_baseline/weights.pt")
    assert registry["pinned_snapshots"] == ["b1_noleague_baseline"]
    assert (good / "training" / "snapshots" / "b1_noleague_baseline" / "weights.pt").read_bytes() == b"weights"


def test_b1_candidate_selection_uses_targeted_confirmation_for_eligibility(tmp_path: Path) -> None:
    run_dir = tmp_path / "confirmed"
    _write_registry(run_dir, (25, 50))
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {
                "update_count": 25,
                "policy_version": 1,
                "aggregate_score": 0.66,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.55,
                    "B3 HeuristicPublicAggro": 0.55,
                    "B4 HeuristicPublicControl": 0.55,
                },
            },
            "train_u50_p2": {
                "update_count": 50,
                "policy_version": 2,
                "aggregate_score": 0.70,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.62,
                    "B3 HeuristicPublicAggro": 0.58,
                    "B4 HeuristicPublicControl": 0.57,
                },
            },
        },
    )
    _write_json(
        run_dir / "eval" / "b1_candidate_confirm64_policy_000001" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 64,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.56},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.54},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.52},
            ],
        },
    )
    _write_json(
        run_dir / "eval" / "b1_candidate_confirm64_policy_000002" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000002",
            "paired_seeds": 64,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.61},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.58},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.49},
            ],
        },
    )

    summary = build_b1_candidate_selection([run_dir])

    selected = summary["selected"]
    assert selected["snapshot_policy_id"] == "policy_000001"
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["required_anchor_min"] == pytest.approx(0.52)
    assert selected["dev_eval_selection_score"] > selected["selection_score"]
    rejected = next(
        candidate for candidate in summary["ranked_candidates"] if candidate["snapshot_policy_id"] == "policy_000002"
    )
    assert rejected["selection_score_source"] == "targeted_confirm"
    assert rejected["eligible"] is False
    assert rejected["ineligibility_reasons"] == ["B4 HeuristicPublicControl 0.4900 < 0.5000"]


def test_b1_candidate_selection_prefers_policy_version_snapshot_over_seed_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "duplicate_updates"
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": "policy_000005",
                    "update": 25,
                    "path": "training/snapshots/policy_000005/weights.pt",
                    "weights_sha256": "live-policy",
                },
                {
                    "policy_id": "seed_7442e4cc88_policy_000001",
                    "update": 25,
                    "path": "training/snapshots/seed_7442e4cc88_policy_000001/weights.pt",
                    "weights_sha256": "seed-import",
                },
            ],
        },
    )
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p5": {
                "update_count": 25,
                "policy_version": 5,
                "aggregate_score": 0.72,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.66,
                    "B3 HeuristicPublicAggro": 0.68,
                    "B4 HeuristicPublicControl": 0.56,
                },
            }
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm64_policy_000005" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000005",
            "paired_seeds": 64,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.59},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.53},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.46},
            ],
        },
    )

    records = load_b1_dev_eval_records(run_dir)
    summary = build_b1_candidate_selection([run_dir])
    selected = summary["selected"]

    assert records[0]["snapshot_policy_id"] == "policy_000005"
    assert records[0]["snapshot_path"] == "training/snapshots/policy_000005/weights.pt"
    assert records[0]["weights_sha256"] == "live-policy"
    assert selected["snapshot_policy_id"] == "policy_000005"
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["selection_confirmation_summary_path"].endswith(
        "targeted_confirm64_policy_000005/targeted_confirm64_summary.json"
    )
    assert selected["eligible"] is False
    assert selected["ineligibility_reasons"] == ["B4 HeuristicPublicControl 0.4600 < 0.5000"]


def test_b1_candidate_selection_prefers_complete_highest_seed_confirmation(tmp_path: Path) -> None:
    run_dir = tmp_path / "confirm_preference"
    _write_registry(run_dir, (25,))
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {
                "update_count": 25,
                "policy_version": 1,
                "aggregate_score": 0.75,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.66,
                    "B3 HeuristicPublicAggro": 0.66,
                    "B4 HeuristicPublicControl": 0.66,
                },
            }
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm256_policy_000001_b4_only" / "targeted_confirm256_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 256,
            "rows": [{"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.40}],
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm64_policy_000001" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 64,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.62},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.60},
            ],
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm128_policy_000001" / "targeted_confirm128_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 128,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.55},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.52},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.50},
            ],
        },
    )

    summary = build_b1_candidate_selection([run_dir])
    selected = summary["selected"]

    assert selected["confirmation"]["paired_seeds"] == 128
    assert selected["required_anchor_min"] == 0.50
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["selection_confirmation_summary_path"].endswith(
        "targeted_confirm128_policy_000001/targeted_confirm128_summary.json"
    )


def test_b1_candidate_selection_ignores_low_seed_confirm_when_higher_confirmation_requested(tmp_path: Path) -> None:
    low_seed = tmp_path / "low_seed"
    high_seed = tmp_path / "high_seed"
    _write_registry(low_seed, (25,))
    _write_registry(high_seed, (25,))
    _write_json(
        low_seed / "eval" / "targeted_confirm64_policy_000001" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 64,
            "mean": 0.80,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.80},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.80},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.80},
            ],
        },
    )
    _write_json(
        high_seed / "eval" / "targeted_confirm256_policy_000001" / "targeted_confirm256_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 256,
            "mean": 0.62,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.66},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.60},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.60},
            ],
        },
    )

    summary = build_b1_candidate_selection([low_seed, high_seed], confirm_paired_seeds=256)
    selected = summary["selected"]

    assert selected["run_name"] == "high_seed"
    assert selected["confirmation"]["paired_seeds"] == 256
    assert summary["candidate_count"] == 1


def test_b1_candidate_selection_prefers_higher_seed_confirm_over_noisier_low_seed_confirm(tmp_path: Path) -> None:
    low_seed_winner = tmp_path / "low_seed_winner"
    high_seed_candidate = tmp_path / "high_seed_candidate"
    _write_registry(low_seed_winner, (25,))
    _write_registry(high_seed_candidate, (25,))
    _write_json(
        low_seed_winner / "eval" / "targeted_confirm64_policy_000001" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 64,
            "mean": 0.88,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.88},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.86},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.84},
            ],
        },
    )
    _write_json(
        high_seed_candidate / "eval" / "targeted_confirm256_policy_000001" / "targeted_confirm256_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 256,
            "mean": 0.61,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.60},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.58},
            ],
        },
    )

    summary = build_b1_candidate_selection([low_seed_winner, high_seed_candidate], confirm_paired_seeds=64)
    selected = summary["selected"]

    assert selected["run_name"] == "high_seed_candidate"
    assert selected["selection_paired_seeds"] == 256
    assert selected["confirmation"]["paired_seeds"] == 256


def test_b1_candidate_selection_prefers_confirmed_candidate_over_periodic_score(tmp_path: Path) -> None:
    run_dir = tmp_path / "confirmed_over_periodic"
    _write_registry(run_dir, (25, 50))
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {
                "update_count": 25,
                "policy_version": 1,
                "aggregate_score": 0.68,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.57,
                    "B3 HeuristicPublicAggro": 0.56,
                    "B4 HeuristicPublicControl": 0.55,
                },
            },
            "train_u50_p2": {
                "update_count": 50,
                "policy_version": 2,
                "aggregate_score": 0.77,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.70,
                    "B3 HeuristicPublicAggro": 0.68,
                    "B4 HeuristicPublicControl": 0.66,
                },
            },
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm64_policy_000001" / "targeted_confirm64_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 64,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.54},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.53},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.52},
            ],
        },
    )

    summary = build_b1_candidate_selection([run_dir])
    selected = summary["selected"]

    assert selected["snapshot_policy_id"] == "policy_000001"
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["selection_score_source_rank"] == 2
    assert selected["eligible"] is True


def test_b1_candidate_selection_includes_targeted_confirm_without_periodic_row(tmp_path: Path) -> None:
    run_dir = tmp_path / "checkpoint_between_periodic_evals"
    _write_registry(run_dir, (25, 90, 100))
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {
                "update_count": 25,
                "policy_version": 1,
                "aggregate_score": 0.60,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.52,
                    "B3 HeuristicPublicAggro": 0.48,
                    "B4 HeuristicPublicControl": 0.51,
                },
            },
            "train_u100_p3": {
                "update_count": 100,
                "policy_version": 3,
                "aggregate_score": 0.58,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.50,
                    "B3 HeuristicPublicAggro": 0.55,
                    "B4 HeuristicPublicControl": 0.42,
                },
            },
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm256_policy_000002" / "targeted_confirm256_summary.json",
        {
            "focal_policy_id": "policy_000002",
            "paired_seeds": 256,
            "mean": 0.66,
            "rows": [
                {"opponent_policy_id": "B0 RandomLegal", "mean": 0.98},
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.58},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.56},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.51},
            ],
        },
    )

    summary = build_b1_candidate_selection([run_dir])
    selected = summary["selected"]

    assert selected["snapshot_policy_id"] == "policy_000002"
    assert selected["train_policy_id"] == "targeted_confirm_policy_000002"
    assert selected["update_count"] == 90
    assert selected["targeted_confirm_only"] is True
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["selection_confirmation_summary_path"].endswith(
        "targeted_confirm256_policy_000002/targeted_confirm256_summary.json"
    )
    assert selected["eligible"] is True
    assert summary["candidate_count"] == 3


def test_b1_candidate_selection_reuses_alias_confirm_when_weights_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "alias_confirm"
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": "policy_000005",
                    "update": 25,
                    "path": "training/snapshots/policy_000005/weights.pt",
                    "weights_sha256": "same-hash",
                },
                {
                    "policy_id": "guided_bootstrap_floor_selected",
                    "update": 25,
                    "path": "training/snapshots/guided_bootstrap_floor_selected/weights.pt",
                    "weights_sha256": "same-hash",
                },
            ],
            "pinned_snapshots": ["guided_bootstrap_floor_selected"],
        },
    )
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p5": {
                "update_count": 25,
                "policy_version": 5,
                "aggregate_score": 0.61,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.52,
                    "B3 HeuristicPublicAggro": 0.52,
                    "B4 HeuristicPublicControl": 0.52,
                },
            }
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm256_guided_alias" / "targeted_confirm256_summary.json",
        {
            "focal_policy_id": "guided_bootstrap_floor_selected",
            "paired_seeds": 256,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.57},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.58},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.55},
            ],
        },
    )

    summary = build_b1_candidate_selection([run_dir])
    selected = summary["selected"]

    assert selected["snapshot_policy_id"] == "policy_000005"
    assert selected["selection_score_source"] == "targeted_confirm"
    assert selected["confirmation"]["paired_seeds"] == 256
    assert selected["confirmation"]["focal_policy_id"] == "guided_bootstrap_floor_selected"
    assert selected["confirmation"]["matched_by_weights_sha256"] is True
    assert selected["selection_confirmation_summary_path"].endswith(
        "targeted_confirm256_guided_alias/targeted_confirm256_summary.json"
    )


def test_publish_selected_candidate_alias_allows_guided_source_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "guided"
    _write_registry(run_dir, (90,))
    source_weights = run_dir / "training" / "snapshots" / "policy_000001" / "weights.pt"
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    source_weights.write_bytes(b"weights")

    published = publish_selected_candidate_alias(
        run_dir=run_dir,
        source_policy_id="policy_000001",
        alias_policy_id="guided_bootstrap_selected",
        selection_summary={"selected": "policy_000001"},
    )

    registry = json.loads((run_dir / "training" / "snapshots" / "registry.json").read_text(encoding="utf-8"))
    snapshots_by_id = {snapshot["policy_id"]: snapshot for snapshot in registry["snapshots"]}
    metadata = json.loads(
        (run_dir / "training" / "snapshots" / "guided_bootstrap_selected" / "policy_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert published["policy_id"] == "guided_bootstrap_selected"
    assert published["alias_for_policy_id"] == "policy_000001"
    assert snapshots_by_id["guided_bootstrap_selected"]["path"] == (
        "training/snapshots/guided_bootstrap_selected/weights.pt"
    )
    assert "guided_bootstrap_selected" in registry["pinned_snapshots"]
    assert metadata["format"] == "selected_candidate_alias_metadata_v1"
    assert metadata["selection_summary"] == {"selected": "policy_000001"}
    assert (run_dir / "training" / "snapshots" / "guided_bootstrap_selected" / "weights.pt").read_bytes() == b"weights"


def test_publish_selected_candidate_alias_is_idempotent_for_same_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "guided"
    _write_registry(run_dir, (90,))
    source_weights = run_dir / "training" / "snapshots" / "policy_000001" / "weights.pt"
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    source_weights.write_bytes(b"weights")
    publish_selected_candidate_alias(
        run_dir=run_dir,
        source_policy_id="policy_000001",
        alias_policy_id="guided_bootstrap_selected",
        selection_summary={"selected": "policy_000001"},
    )

    refreshed = publish_selected_candidate_alias(
        run_dir=run_dir,
        source_policy_id="guided_bootstrap_selected",
        alias_policy_id="guided_bootstrap_selected",
        selection_summary={"selected": "guided_bootstrap_selected", "confirm": 128},
    )

    metadata = json.loads(
        (run_dir / "training" / "snapshots" / "guided_bootstrap_selected" / "policy_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert refreshed["policy_id"] == "guided_bootstrap_selected"
    assert refreshed["alias_for_policy_id"] == "guided_bootstrap_selected"
    assert metadata["alias_for_policy_id"] == "guided_bootstrap_selected"
    assert metadata["selection_summary"]["confirm"] == 128
    assert (run_dir / "training" / "snapshots" / "guided_bootstrap_selected" / "weights.pt").read_bytes() == b"weights"


def test_publish_selected_candidate_alias_refuses_canonical_b1_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "guided"
    _write_registry(run_dir, (90,))

    with pytest.raises(ValueError, match="canonical B1 baseline"):
        publish_selected_candidate_alias(
            run_dir=run_dir,
            source_policy_id="policy_000001",
            alias_policy_id="b1_noleague_baseline",
        )


def test_b1_candidate_selection_marks_no_eligible_candidate(tmp_path: Path) -> None:
    run_dir = tmp_path / "weak"
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": "b1_noleague_baseline",
                    "update": 50,
                    "path": "training/snapshots/b1_noleague_baseline/weights.pt",
                    "weights_sha256": "auto-alias",
                }
            ],
        },
    )
    _write_json(
        run_dir / "training" / "snapshots" / "b1_noleague_baseline" / "policy_meta.json",
        {
            "format": "minimal_train_snapshot_metadata_v1",
            "policy_id": "b1_noleague_baseline",
            "update": 50,
        },
    )
    _write_json(
        run_dir / "eval" / "dev_eval" / "update_25" / "summary.json",
        {
            "policy_id": "train_u25_p1",
            "update_count": 25,
            "aggregate_score": 0.45,
            "anchor_scores": {"B2 HeuristicPublic": 0.25},
        },
    )

    records = load_b1_dev_eval_records(run_dir)
    summary = build_b1_candidate_selection([run_dir])

    assert records[0]["snapshot_policy_id"] == "train_u25_p1"
    assert summary["selected"]["eligible"] is False
    assert summary["run_summaries"][0]["baseline_alias"]["metadata_format"] == "minimal_train_snapshot_metadata_v1"
    assert any("non-selector b1_noleague_baseline alias" in warning for warning in summary["warnings"])
    assert "no B1 candidate met the required anchor threshold" in summary["warnings"]


def test_publish_b1_baseline_alias_refuses_guided_source_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "guided"
    _write_registry(run_dir, (75,))
    source_weights = run_dir / "training" / "snapshots" / "policy_000001" / "weights.pt"
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    source_weights.write_bytes(b"weights")
    _write_json(
        run_dir / "manifest.json", {"config_canonical": {"config": {"experiment": {"role": "ablation_guided"}}}}
    )

    with pytest.raises(RuntimeError, match="not marked experiment.role='baseline_noleague'"):
        publish_b1_baseline_alias(run_dir=run_dir, source_policy_id="policy_000001")


def test_load_reference_anchor_scores_reads_targeted_confirm_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "b2_vs_b3b4" / "targeted_confirm128_summary.json"
    _write_json(
        summary_path,
        {
            "elapsed_seconds": 27.0,
            "focal_policy_id": "B2 HeuristicPublic",
            "paired_seeds": 128,
            "rows": [
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.31},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.47},
            ],
        },
    )

    assert load_reference_anchor_scores(summary_path) == {
        "B3 HeuristicPublicAggro": 0.31,
        "B4 HeuristicPublicControl": 0.47,
    }


def test_b1_candidate_selection_reports_reference_deltas(tmp_path: Path) -> None:
    run_dir = tmp_path / "confirmed"
    _write_registry(run_dir, (50,))
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u50_p1": {
                "update_count": 50,
                "policy_version": 1,
                "aggregate_score": 0.70,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.61,
                    "B3 HeuristicPublicAggro": 0.55,
                    "B4 HeuristicPublicControl": 0.48,
                },
            }
        },
    )
    _write_json(
        run_dir / "eval" / "targeted_confirm128_policy_000001" / "targeted_confirm128_summary.json",
        {
            "focal_policy_id": "policy_000001",
            "paired_seeds": 128,
            "rows": [
                {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.62},
                {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.56},
                {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.49},
            ],
        },
    )

    summary = build_b1_candidate_selection(
        [run_dir],
        reference_anchor_scores={
            "B3 HeuristicPublicAggro": 0.32,
            "B4 HeuristicPublicControl": 0.47,
        },
        reference_label="B2 anchor reference",
    )

    comparison = summary["selected"]["reference_comparison"]
    assert summary["reference_label"] == "B2 anchor reference"
    assert summary["reference_anchor_scores"] == {
        "B3 HeuristicPublicAggro": 0.32,
        "B4 HeuristicPublicControl": 0.47,
    }
    assert comparison["reference_label"] == "B2 anchor reference"
    assert comparison["common_anchors"] == ["B3 HeuristicPublicAggro", "B4 HeuristicPublicControl"]
    assert comparison["anchor_deltas"]["B3 HeuristicPublicAggro"] == pytest.approx(0.24)
    assert comparison["anchor_deltas"]["B4 HeuristicPublicControl"] == pytest.approx(0.02)
    assert comparison["min_delta"] == pytest.approx(0.02)
    assert comparison["all_common_at_or_above_reference"] is True
