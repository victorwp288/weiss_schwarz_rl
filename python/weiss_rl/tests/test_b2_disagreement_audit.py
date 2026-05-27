from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.heuristic_public import ActionCatalog

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "python" / "scripts" / "b2_disagreement_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("b2_disagreement_audit_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_record(
    *,
    pair_index: int,
    swap_index: int,
    episode_seed: int,
    focal_policy_id: str = "learner",
    opponent_policy_id: str = "B2 HeuristicPublic",
) -> EvalGameRecord:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=episode_seed,
        episode_key=f"episode-{pair_index}-{swap_index}",
        episode_key64=pair_index * 10 + swap_index,
        config_hash256="a" * 64,
        spec_hash256="b" * 64,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome="W" if focal_seat == 0 else "L",
        terminated=True,
        truncated=False,
        engine_status=0,
        decision_count=1,
        tick_count=1,
        no_progress_count=0,
        termination_reason="terminated",
    )


def test_load_matchup_source_extracts_unique_paired_seeds_in_first_seen_order(tmp_path: Path) -> None:
    module = _load_script_module()
    episodes_path = tmp_path / "episodes.jsonl"
    records = [
        _make_record(pair_index=2, swap_index=0, episode_seed=42),
        _make_record(pair_index=2, swap_index=1, episode_seed=42),
        _make_record(pair_index=0, swap_index=0, episode_seed=7),
        _make_record(pair_index=0, swap_index=1, episode_seed=7),
        _make_record(pair_index=1, swap_index=0, episode_seed=42),
        _make_record(pair_index=1, swap_index=1, episode_seed=42),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    source = module._load_matchup_source(episodes_path)

    assert source.focal_policy_id == "learner"
    assert source.opponent_policy_id == "B2 HeuristicPublic"
    assert source.paired_seeds == (42, 7)


def test_b2_disagreement_audit_requires_fixed_pythonhashseed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/b2_disagreement_audit.py",
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--output-run-dir",
            str(tmp_path / "audit"),
            "--episodes-jsonl",
            "missing_episodes.jsonl",
            "--policy-id",
            "policy_000001",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_b2_disagreement_audit_parser_accepts_b1_baseline_run_dir() -> None:
    module = _load_script_module()

    args = module._build_parser().parse_args(
        [
            "--stack-config",
            "configs/thesis/main.yaml",
            "--run-dir",
            "runs/main",
            "--output-run-dir",
            "runs/audit",
            "--episodes-jsonl",
            "runs/main/eval/b1/episodes.jsonl",
            "--policy-id",
            "policy_000002",
            "--opponent-policy-id",
            "B1 NoLeague baseline",
            "--b1-baseline-run-dir",
            "runs/locked_b1",
            "--require-opponent-context-index",
        ]
    )

    assert args.b1_baseline_run_dir == Path("runs/locked_b1")
    assert args.require_opponent_context_index is True


def test_resolve_requested_policy_id_accepts_registry_alias_for_train_policy_id() -> None:
    module = _load_script_module()

    resolved = module._resolve_requested_policy_id(
        requested_policy_id="policy_000015",
        source_focal_policy_id="train_u300_p15",
    )

    assert resolved == "policy_000015"


def test_inspection_policy_id_maps_b1_display_id_to_registry_alias() -> None:
    module = _load_script_module()

    assert module._inspection_policy_id("B1 NoLeague baseline") == "b1_noleague_baseline"
    assert module._inspection_policy_id("B2 HeuristicPublic") == "B2 HeuristicPublic"


def test_run_config_hashes_reads_b1_hash_from_hash_file_and_manifest(tmp_path: Path) -> None:
    module = _load_script_module()
    run_dir = tmp_path / "b1"
    run_dir.mkdir()
    run_dir.joinpath("config_hash256.txt").write_text("a" * 64 + "\n", encoding="utf-8")
    run_dir.joinpath("manifest.json").write_text(json.dumps({"config_hash256": "b" * 64}), encoding="utf-8")

    assert module._run_config_hashes(run_dir) == ["a" * 64, "b" * 64]


def test_resolve_requested_policy_id_requires_explicit_mismatch_mode() -> None:
    module = _load_script_module()

    rejected = module._resolve_requested_policy_id(
        requested_policy_id="selected_seed",
        source_focal_policy_id="policy_000005",
    )
    accepted = module._resolve_requested_policy_id(
        requested_policy_id="selected_seed",
        source_focal_policy_id="policy_000005",
        allow_mismatch=True,
    )

    assert rejected is None
    assert accepted == "selected_seed"


def test_resolve_source_config_hash_accepts_run_manifest_hash(tmp_path: Path) -> None:
    module = _load_script_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.joinpath("manifest.json").write_text(
        json.dumps({"config_hash256": "b" * 64}),
        encoding="utf-8",
    )

    resolved, manifest_hash = module._resolve_source_config_hash(
        source_config_hash256="b" * 64,
        stack_config_hash256="a" * 64,
        run_dir=run_dir,
    )

    assert resolved == "b" * 64
    assert manifest_hash == "b" * 64


def test_resolve_source_config_hash_rejects_unmatched_hash(tmp_path: Path) -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="stack config hash does not match"):
        module._resolve_source_config_hash(
            source_config_hash256="c" * 64,
            stack_config_hash256="a" * 64,
            run_dir=tmp_path / "missing_run",
        )


def test_annotate_step_diff_uses_action_catalog_decoder() -> None:
    module = _load_script_module()
    decoder = ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_space_size": 7,
                "pass_action_id": 0,
                "constants": [["MAX_HAND", 50], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "attack_type_encoding": [["front", 0], ["side", 1]],
                "families": [
                    {"name": "pass", "base": 0, "count": 1},
                    {"name": "attack", "base": 1, "count": 6},
                ],
            }
        }
    )

    annotated = module._annotate_step_diff(
        {
            "recorded_action": 0,
            "policy_a_top_action": {"action": 0},
            "policy_b_top_action": {"action": 1},
        },
        decoder=decoder,
    )

    assert annotated["recorded_action_family"] == "pass"
    assert annotated["policy_a_top_action_family"] == "pass"
    assert annotated["policy_b_top_action_family"] == "attack"


def test_aggregate_audit_summary_ranks_repeated_family_pairs_and_weighted_means(tmp_path: Path) -> None:
    module = _load_script_module()
    source = module.MatchupSource(
        focal_policy_id="learner",
        opponent_policy_id="B2 HeuristicPublic",
        config_hash256="a" * 64,
        spec_hash256="b" * 64,
        paired_seeds=(42, 7),
    )
    bundle_summaries = [
        {
            "bundle_path": "/tmp/bundle-1.zip",
            "report_path": "/tmp/report-1.json",
            "pair_index": 0,
            "swap_index": 0,
            "episode_seed": 42,
            "replay_key64": "1111",
            "summary": {
                "compared_steps": 10,
                "max_total_variation": 0.6,
                "mean_total_variation": 0.2,
                "policy_a_matches_policy_b_top_action_rate": 0.1,
                "policy_a_matches_policy_b_top_action_family_rate": 0.2,
                "policy_a_mean_probability_on_policy_b_top_action": 0.3,
                "policy_a_mean_probability_on_policy_b_top_action_family": 0.4,
                "policy_a_median_rank_of_policy_b_top_action": 2.0,
                "policy_a_legal_surface_filter_rate": 0.7,
                "policy_b_legal_surface_filter_rate": 0.0,
                "policy_a_mean_raw_minus_policy_a_legal_action_count": 2.0,
                "policy_b_mean_raw_minus_policy_b_legal_action_count": 0.0,
                "policy_b_top_action_illegal_for_policy_a_rate": 0.6,
                "policy_a_top_action_illegal_for_policy_b_rate": 0.0,
                "policy_a_probability_on_policy_b_top_action_percentiles": {
                    "count": 10,
                    "mean": 0.3,
                    "p10": 0.1,
                    "p25": 0.2,
                    "p50": 0.3,
                    "p75": 0.4,
                    "p90": 0.5,
                },
                "policy_a_top_logit_margin_percentiles": {
                    "count": 10,
                    "mean": 0.2,
                    "p10": 0.05,
                    "p25": 0.1,
                    "p50": 0.2,
                    "p75": 0.3,
                    "p90": 0.4,
                },
                "policy_a_top_probability_margin_percentiles": {
                    "count": 10,
                    "mean": 0.05,
                    "p10": 0.01,
                    "p25": 0.02,
                    "p50": 0.05,
                    "p75": 0.08,
                    "p90": 0.1,
                },
                "policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles": {
                    "count": 10,
                    "mean": 0.8,
                    "p10": 0.1,
                    "p25": 0.3,
                    "p50": 0.8,
                    "p75": 1.2,
                    "p90": 1.5,
                },
                "raw_legal_action_count_percentiles": {
                    "count": 10,
                    "mean": 6.0,
                    "p10": 2.0,
                    "p25": 4.0,
                    "p50": 6.0,
                    "p75": 8.0,
                    "p90": 10.0,
                },
                "policy_a_legal_action_count_percentiles": {
                    "count": 10,
                    "mean": 4.0,
                    "p10": 1.0,
                    "p25": 2.0,
                    "p50": 4.0,
                    "p75": 6.0,
                    "p90": 8.0,
                },
                "policy_b_legal_action_count_percentiles": {
                    "count": 10,
                    "mean": 6.0,
                    "p10": 2.0,
                    "p25": 4.0,
                    "p50": 6.0,
                    "p75": 8.0,
                    "p90": 10.0,
                },
                "top_action_family_confusions": [
                    {"policy_b_family": "pass", "policy_a_family": "attack", "count": 6},
                    {"policy_b_family": "attack", "policy_a_family": "attack", "count": 4},
                ],
                "policy_a_mean_family_probability_masses": [
                    {"family": "attack", "mean_probability": 0.6},
                    {"family": "pass", "mean_probability": 0.4},
                ],
                "policy_b_top_family_summaries": [
                    {
                        "family": "attack",
                        "count": 4,
                        "policy_a_matches_policy_b_top_action_rate": 0.25,
                        "policy_a_matches_policy_b_top_action_family_rate": 0.5,
                        "policy_a_mean_probability_on_policy_b_top_action": 0.4,
                        "policy_a_mean_probability_on_policy_b_top_action_family": 0.8,
                        "policy_b_top_action_legal_for_policy_a_rate": 0.5,
                        "policy_a_legal_surface_filter_rate": 0.25,
                        "policy_a_mean_raw_minus_policy_a_legal_action_count": 1.0,
                        "policy_a_probability_on_policy_b_top_action_percentiles": {
                            "count": 4,
                            "mean": 0.4,
                            "p10": 0.1,
                            "p25": 0.2,
                            "p50": 0.4,
                            "p75": 0.6,
                            "p90": 0.7,
                        },
                        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": {
                            "count": 4,
                            "mean": 0.1,
                            "p10": -0.1,
                            "p25": 0.0,
                            "p50": 0.1,
                            "p75": 0.2,
                            "p90": 0.3,
                        },
                    },
                    {
                        "family": "pass",
                        "count": 6,
                        "policy_a_matches_policy_b_top_action_rate": 0.0,
                        "policy_a_matches_policy_b_top_action_family_rate": 0.0,
                        "policy_a_mean_probability_on_policy_b_top_action": 0.2,
                        "policy_a_mean_probability_on_policy_b_top_action_family": 0.2,
                        "policy_b_top_action_legal_for_policy_a_rate": 0.0,
                        "policy_a_legal_surface_filter_rate": 1.0,
                        "policy_a_mean_raw_minus_policy_a_legal_action_count": 3.0,
                        "policy_a_probability_on_policy_b_top_action_percentiles": {
                            "count": 6,
                            "mean": 0.2,
                            "p10": 0.1,
                            "p25": 0.15,
                            "p50": 0.2,
                            "p75": 0.25,
                            "p90": 0.3,
                        },
                        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": {
                            "count": 0,
                            "mean": None,
                            "p10": None,
                            "p25": None,
                            "p50": None,
                            "p75": None,
                            "p90": None,
                        },
                    },
                ],
            },
            "compared_steps": 10,
            "inspected_step_count": 4,
            "family_pair_counts": [
                {"policy_a_family": "attack", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "attack", "policy_b_family": "attack", "count": 1},
                {"policy_a_family": "clock_from_hand", "policy_b_family": "clock_from_hand", "count": 1},
            ],
            "policy_a_family_counts": [
                {"family": "attack", "count": 3},
                {"family": "clock_from_hand", "count": 1},
            ],
            "policy_b_family_counts": [
                {"family": "pass", "count": 2},
                {"family": "attack", "count": 1},
                {"family": "clock_from_hand", "count": 1},
            ],
            "recorded_family_counts": [
                {"family": "attack", "count": 2},
                {"family": "pass", "count": 1},
                {"family": "clock_from_hand", "count": 1},
            ],
            "action_label_pair_counts": [
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "pass",
                    "count": 2,
                },
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "attack(slot=0, attack_type=direct)",
                    "count": 1,
                },
                {
                    "policy_a_action_label": "clock_from_hand(hand_index=0)",
                    "policy_b_action_label": "clock_from_hand(hand_index=0)",
                    "count": 1,
                },
            ],
            "policy_a_action_label_counts": [
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 3},
                {"action_label": "clock_from_hand(hand_index=0)", "count": 1},
            ],
            "policy_b_action_label_counts": [
                {"action_label": "pass", "count": 2},
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 1},
                {"action_label": "clock_from_hand(hand_index=0)", "count": 1},
            ],
            "top_examples": [{"total_variation": 0.4, "example": "first"}],
        },
        {
            "bundle_path": "/tmp/bundle-2.zip",
            "report_path": "/tmp/report-2.json",
            "pair_index": 1,
            "swap_index": 1,
            "episode_seed": 7,
            "replay_key64": "2222",
            "summary": {
                "compared_steps": 20,
                "max_total_variation": 0.9,
                "mean_total_variation": 0.4,
                "policy_a_matches_policy_b_top_action_rate": 0.4,
                "policy_a_matches_policy_b_top_action_family_rate": 0.5,
                "policy_a_mean_probability_on_policy_b_top_action": 0.6,
                "policy_a_mean_probability_on_policy_b_top_action_family": 0.7,
                "policy_a_median_rank_of_policy_b_top_action": 4.0,
                "policy_a_legal_surface_filter_rate": 0.2,
                "policy_b_legal_surface_filter_rate": 0.1,
                "policy_a_mean_raw_minus_policy_a_legal_action_count": 0.5,
                "policy_b_mean_raw_minus_policy_b_legal_action_count": 0.25,
                "policy_b_top_action_illegal_for_policy_a_rate": 0.15,
                "policy_a_top_action_illegal_for_policy_b_rate": 0.05,
                "policy_a_probability_on_policy_b_top_action_percentiles": {
                    "count": 20,
                    "mean": 0.6,
                    "p10": 0.2,
                    "p25": 0.4,
                    "p50": 0.65,
                    "p75": 0.8,
                    "p90": 0.9,
                },
                "policy_a_top_logit_margin_percentiles": {
                    "count": 20,
                    "mean": 0.5,
                    "p10": 0.1,
                    "p25": 0.3,
                    "p50": 0.6,
                    "p75": 0.8,
                    "p90": 1.0,
                },
                "policy_a_top_probability_margin_percentiles": {
                    "count": 20,
                    "mean": 0.12,
                    "p10": 0.02,
                    "p25": 0.06,
                    "p50": 0.12,
                    "p75": 0.18,
                    "p90": 0.24,
                },
                "policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles": {
                    "count": 20,
                    "mean": 0.3,
                    "p10": 0.0,
                    "p25": 0.1,
                    "p50": 0.2,
                    "p75": 0.5,
                    "p90": 0.7,
                },
                "raw_legal_action_count_percentiles": {
                    "count": 20,
                    "mean": 5.0,
                    "p10": 2.0,
                    "p25": 3.0,
                    "p50": 5.0,
                    "p75": 7.0,
                    "p90": 9.0,
                },
                "policy_a_legal_action_count_percentiles": {
                    "count": 20,
                    "mean": 4.5,
                    "p10": 2.0,
                    "p25": 3.0,
                    "p50": 4.0,
                    "p75": 6.0,
                    "p90": 8.0,
                },
                "policy_b_legal_action_count_percentiles": {
                    "count": 20,
                    "mean": 4.75,
                    "p10": 2.0,
                    "p25": 3.0,
                    "p50": 5.0,
                    "p75": 7.0,
                    "p90": 9.0,
                },
                "top_action_family_confusions": [
                    {"policy_b_family": "pass", "policy_a_family": "main_move", "count": 12},
                    {"policy_b_family": "attack", "policy_a_family": "attack", "count": 8},
                ],
                "policy_a_mean_family_probability_masses": [
                    {"family": "attack", "mean_probability": 0.2},
                    {"family": "pass", "mean_probability": 0.8},
                ],
                "policy_b_top_family_summaries": [
                    {
                        "family": "attack",
                        "count": 8,
                        "policy_a_matches_policy_b_top_action_rate": 0.5,
                        "policy_a_matches_policy_b_top_action_family_rate": 0.75,
                        "policy_a_mean_probability_on_policy_b_top_action": 0.7,
                        "policy_a_mean_probability_on_policy_b_top_action_family": 0.9,
                        "policy_b_top_action_legal_for_policy_a_rate": 0.875,
                        "policy_a_legal_surface_filter_rate": 0.125,
                        "policy_a_mean_raw_minus_policy_a_legal_action_count": 0.5,
                        "policy_a_probability_on_policy_b_top_action_percentiles": {
                            "count": 8,
                            "mean": 0.7,
                            "p10": 0.5,
                            "p25": 0.6,
                            "p50": 0.7,
                            "p75": 0.8,
                            "p90": 0.9,
                        },
                        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": {
                            "count": 8,
                            "mean": 0.4,
                            "p10": 0.1,
                            "p25": 0.2,
                            "p50": 0.4,
                            "p75": 0.6,
                            "p90": 0.8,
                        },
                    },
                    {
                        "family": "pass",
                        "count": 12,
                        "policy_a_matches_policy_b_top_action_rate": 0.25,
                        "policy_a_matches_policy_b_top_action_family_rate": 0.25,
                        "policy_a_mean_probability_on_policy_b_top_action": 0.5,
                        "policy_a_mean_probability_on_policy_b_top_action_family": 0.5,
                        "policy_b_top_action_legal_for_policy_a_rate": 0.75,
                        "policy_a_legal_surface_filter_rate": 0.25,
                        "policy_a_mean_raw_minus_policy_a_legal_action_count": 0.5,
                        "policy_a_probability_on_policy_b_top_action_percentiles": {
                            "count": 12,
                            "mean": 0.5,
                            "p10": 0.3,
                            "p25": 0.4,
                            "p50": 0.5,
                            "p75": 0.6,
                            "p90": 0.7,
                        },
                        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": {
                            "count": 0,
                            "mean": None,
                            "p10": None,
                            "p25": None,
                            "p50": None,
                            "p75": None,
                            "p90": None,
                        },
                    },
                ],
            },
            "compared_steps": 20,
            "inspected_step_count": 5,
            "family_pair_counts": [
                {"policy_a_family": "attack", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "attack", "policy_b_family": "attack", "count": 1},
            ],
            "policy_a_family_counts": [
                {"family": "attack", "count": 3},
                {"family": "main_move", "count": 2},
            ],
            "policy_b_family_counts": [
                {"family": "pass", "count": 4},
                {"family": "attack", "count": 1},
            ],
            "recorded_family_counts": [
                {"family": "main_move", "count": 2},
                {"family": "attack", "count": 2},
                {"family": "pass", "count": 1},
            ],
            "action_label_pair_counts": [
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "pass",
                    "count": 2,
                },
                {
                    "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                    "policy_b_action_label": "pass",
                    "count": 2,
                },
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "attack(slot=0, attack_type=direct)",
                    "count": 1,
                },
            ],
            "policy_a_action_label_counts": [
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 3},
                {"action_label": "main_move(from_slot=0, to_slot=2)", "count": 2},
            ],
            "policy_b_action_label_counts": [
                {"action_label": "pass", "count": 4},
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 1},
            ],
            "top_examples": [{"total_variation": 0.8, "example": "second"}],
        },
    ]

    summary = module._aggregate_audit_summary(
        source=source,
        policy_id="learner",
        opponent_policy_id=source.opponent_policy_id,
        episodes_jsonl=tmp_path / "source.jsonl",
        run_dir=tmp_path / "source_run",
        output_run_dir=tmp_path / "output_run",
        episodes_path=tmp_path / "audit" / "episodes.jsonl",
        game_count=4,
        bundle_summaries=bundle_summaries,
        inspection_errors=[],
    )

    assert summary["status"] == "ok"
    assert summary["opponent_policy_id"] == "B2 HeuristicPublic"
    assert summary["games"] == 4
    assert summary["bundle_count"] == 2
    assert summary["top_family_pairs"][0] == {
        "policy_a_family": "attack",
        "policy_b_family": "pass",
        "count": 4,
    }
    assert summary["top_policy_a_families"][0] == {"family": "attack", "count": 6}
    assert summary["top_policy_b_families"][0] == {"family": "pass", "count": 6}
    assert summary["top_recorded_families"][0] == {"family": "attack", "count": 4}
    assert summary["top_action_label_pairs"][0] == {
        "policy_a_action_label": "attack(slot=0, attack_type=direct)",
        "policy_b_action_label": "pass",
        "count": 4,
    }
    assert summary["top_action_family_confusions"][:3] == [
        {"policy_b_family": "attack", "policy_a_family": "attack", "count": 12},
        {"policy_b_family": "pass", "policy_a_family": "main_move", "count": 12},
        {"policy_b_family": "pass", "policy_a_family": "attack", "count": 6},
    ]
    assert summary["top_policy_a_action_labels"][0] == {
        "action_label": "attack(slot=0, attack_type=direct)",
        "count": 6,
    }
    assert summary["top_policy_b_action_labels"][0] == {"action_label": "pass", "count": 6}
    assert summary["max_total_variation"] == pytest.approx(0.9)
    assert summary["mean_total_variation"] == pytest.approx((0.2 * 10 + 0.4 * 20) / 30)
    assert summary["policy_a_matches_policy_b_top_action_rate"] == pytest.approx((0.1 * 10 + 0.4 * 20) / 30)
    assert summary["policy_a_matches_policy_b_top_action_family_rate"] == pytest.approx((0.2 * 10 + 0.5 * 20) / 30)
    assert summary["policy_a_mean_probability_on_policy_b_top_action"] == pytest.approx((0.3 * 10 + 0.6 * 20) / 30)
    assert summary["policy_a_mean_probability_on_policy_b_top_action_family"] == pytest.approx(
        (0.4 * 10 + 0.7 * 20) / 30
    )
    assert summary["policy_a_weighted_mean_median_rank_of_policy_b_top_action"] == pytest.approx(
        (2.0 * 10 + 4.0 * 20) / 30
    )
    assert summary["policy_a_legal_surface_filter_rate"] == pytest.approx((0.7 * 10 + 0.2 * 20) / 30)
    assert summary["policy_b_legal_surface_filter_rate"] == pytest.approx((0.0 * 10 + 0.1 * 20) / 30)
    assert summary["policy_a_mean_raw_minus_policy_a_legal_action_count"] == pytest.approx((2.0 * 10 + 0.5 * 20) / 30)
    assert summary["policy_b_mean_raw_minus_policy_b_legal_action_count"] == pytest.approx((0.0 * 10 + 0.25 * 20) / 30)
    assert summary["policy_b_top_action_illegal_for_policy_a_rate"] == pytest.approx((0.6 * 10 + 0.15 * 20) / 30)
    assert summary["policy_a_top_action_illegal_for_policy_b_rate"] == pytest.approx((0.0 * 10 + 0.05 * 20) / 30)
    assert summary["policy_a_top_logit_margin_percentiles_bundle_weighted"] == {
        "aggregation": "weighted_mean_of_bundle_percentiles",
        "source_summary_key": "policy_a_top_logit_margin_percentiles",
        "count": 30,
        "mean": pytest.approx((0.2 * 10 + 0.5 * 20) / 30),
        "p10": pytest.approx((0.05 * 10 + 0.1 * 20) / 30),
        "p25": pytest.approx((0.1 * 10 + 0.3 * 20) / 30),
        "p50": pytest.approx((0.2 * 10 + 0.6 * 20) / 30),
        "p75": pytest.approx((0.3 * 10 + 0.8 * 20) / 30),
        "p90": pytest.approx((0.4 * 10 + 1.0 * 20) / 30),
    }
    assert summary["policy_a_probability_on_policy_b_top_action_percentiles_bundle_weighted"]["p50"] == pytest.approx(
        (0.3 * 10 + 0.65 * 20) / 30
    )
    assert summary["policy_a_top_probability_margin_percentiles_bundle_weighted"]["mean"] == pytest.approx(
        (0.05 * 10 + 0.12 * 20) / 30
    )
    assert summary["policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles_bundle_weighted"][
        "p90"
    ] == pytest.approx((1.5 * 10 + 0.7 * 20) / 30)
    assert summary["raw_legal_action_count_percentiles_bundle_weighted"]["mean"] == pytest.approx(
        (6.0 * 10 + 5.0 * 20) / 30
    )
    assert summary["policy_a_legal_action_count_percentiles_bundle_weighted"]["p50"] == pytest.approx(
        (4.0 * 10 + 4.0 * 20) / 30
    )
    assert summary["policy_a_policy_b_top_action_same_family_logit_margin_percentiles_bundle_weighted"]["count"] == 0
    assert summary["policy_b_top_family_summaries"][0]["family"] == "pass"
    assert summary["policy_b_top_family_summaries"][0]["count"] == 18
    assert summary["policy_b_top_family_summaries"][1]["family"] == "attack"
    assert summary["policy_b_top_family_summaries"][1]["count"] == 12
    assert summary["policy_b_top_family_summaries"][1]["policy_a_matches_policy_b_top_action_rate"] == pytest.approx(
        (0.25 * 4 + 0.5 * 8) / 12
    )
    assert summary["policy_b_top_family_summaries"][1][
        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles_bundle_weighted"
    ]["mean"] == pytest.approx((0.1 * 4 + 0.4 * 8) / 12)
    assert summary["policy_b_top_family_summaries"][1]["policy_b_top_action_legal_for_policy_a_rate"] == pytest.approx(
        (0.5 * 4 + 0.875 * 8) / 12
    )
    assert summary["policy_b_top_family_summaries"][1]["policy_a_legal_surface_filter_rate"] == pytest.approx(
        (0.25 * 4 + 0.125 * 8) / 12
    )
    assert summary["policy_a_mean_family_probability_masses"][0] == {
        "family": "pass",
        "mean_probability": pytest.approx((0.4 * 10 + 0.8 * 20) / 30),
    }
    assert summary["top_examples"][0]["example"] == "second"


def test_audit_run_id_includes_opponent_policy_id(tmp_path: Path) -> None:
    module = _load_script_module()

    b2_id = module._audit_run_id256(
        policy_id="policy_000002",
        opponent_policy_id="B2 HeuristicPublic",
        episodes_jsonl=tmp_path / "episodes.jsonl",
        output_run_dir=tmp_path / "out",
        paired_seeds=(1, 2),
    )
    b3_id = module._audit_run_id256(
        policy_id="policy_000002",
        opponent_policy_id="B3 HeuristicPublicAggro",
        episodes_jsonl=tmp_path / "episodes.jsonl",
        output_run_dir=tmp_path / "out",
        paired_seeds=(1, 2),
    )

    assert b2_id != b3_id


def test_aggregate_trajectory_summary_merges_counts_and_focal_roles() -> None:
    module = _load_script_module()

    bundle_summaries = [
        {
            "focal_seat": 0,
            "trajectory_summary": {
                "compared_steps": 2,
                "recorded_family_counts": [{"family": "pass", "count": 1}, {"family": "attack", "count": 1}],
                "phase_counts": [{"phase": "2", "count": 2}],
                "decision_kind_counts": [{"decision_kind": "3", "count": 2}],
                "legal_family_presence_rates": [{"family": "attack", "rate": 0.5}],
                "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=3.0)},
                "actor_summaries": [
                    {
                        "actor": 0,
                        "compared_steps": 1,
                        "recorded_family_counts": [{"family": "pass", "count": 1}],
                        "phase_counts": [{"phase": "2", "count": 1}],
                        "decision_kind_counts": [{"decision_kind": "3", "count": 1}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 0.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=1, mean=2.0)},
                    },
                    {
                        "actor": 1,
                        "compared_steps": 1,
                        "recorded_family_counts": [{"family": "attack", "count": 1}],
                        "phase_counts": [{"phase": "2", "count": 1}],
                        "decision_kind_counts": [{"decision_kind": "3", "count": 1}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=1, mean=5.0)},
                    },
                ],
            },
        },
        {
            "focal_seat": 1,
            "trajectory_summary": {
                "compared_steps": 2,
                "recorded_family_counts": [{"family": "clock_from_hand", "count": 2}],
                "phase_counts": [{"phase": "1", "count": 2}],
                "decision_kind_counts": [{"decision_kind": "4", "count": 2}],
                "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=4.0)},
                "actor_summaries": [
                    {
                        "actor": 1,
                        "compared_steps": 2,
                        "recorded_family_counts": [{"family": "clock_from_hand", "count": 2}],
                        "phase_counts": [{"phase": "1", "count": 2}],
                        "decision_kind_counts": [{"decision_kind": "4", "count": 2}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=4.0)},
                    }
                ],
            },
        },
    ]

    summary = module._aggregate_trajectory_summary(bundle_summaries)

    assert summary["compared_steps"] == 4
    assert summary["recorded_family_counts"] == [
        {"family": "clock_from_hand", "count": 2},
        {"family": "attack", "count": 1},
        {"family": "pass", "count": 1},
    ]
    assert summary["numeric_summaries"]["self_clock_count"]["mean"] == pytest.approx((3.0 * 2 + 4.0 * 2) / 4)
    assert summary["legal_family_presence_rates"] == [{"family": "attack", "rate": 0.75}]
    role_summaries = {item["role"]: item for item in summary["role_summaries"]}
    assert role_summaries["focal"]["compared_steps"] == 3
    assert role_summaries["focal"]["recorded_family_counts"][0] == {"family": "clock_from_hand", "count": 2}
    assert role_summaries["opponent"]["compared_steps"] == 1
    assert role_summaries["opponent"]["recorded_family_counts"][0] == {"family": "attack", "count": 1}


def test_materialize_audit_bundle_copy_uses_pair_and_swap_suffix(tmp_path: Path) -> None:
    module = _load_script_module()
    source_bundle = tmp_path / "replays" / "bundles" / "replay_deadbeef.zip"
    source_bundle.parent.mkdir(parents=True, exist_ok=True)
    source_bundle.write_text("bundle-bytes", encoding="utf-8")
    bundle_copies_dir = tmp_path / "audit" / "replay_bundles"
    bundle_copies_dir.mkdir(parents=True, exist_ok=True)

    copied = module._materialize_audit_bundle_copy(
        source_bundle_path=source_bundle,
        bundle_copies_dir=bundle_copies_dir,
        pair_index=7,
        swap_index=1,
    )

    assert copied == bundle_copies_dir / "replay_deadbeef_pair007_swap1.zip"
    assert copied.read_text(encoding="utf-8") == "bundle-bytes"


def _numeric_summary(*, count: int, mean: float) -> dict[str, float | int]:
    return {
        "count": int(count),
        "mean": float(mean),
        "p10": float(mean),
        "p25": float(mean),
        "p50": float(mean),
        "p75": float(mean),
        "p90": float(mean),
    }
