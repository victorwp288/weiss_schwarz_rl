from __future__ import annotations

import os
from pathlib import Path

from weiss_rl.eval.simulator.harness import EvalGameRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"]
    return env


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


def weighted_audit_bundle_summaries() -> list[dict[str, object]]:
    return [
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
