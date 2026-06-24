from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from weiss_rl.config.sections.sections_evaluation import parse_evaluation_config


def _copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def _evaluation_body() -> dict[str, object]:
    return {
        "seat_swap": True,
        "eval_device": "cpu",
        "eval_inference_mode": True,
        "eval_sampling_algorithm": "deterministic_argmax",
        "eval_assert_sorted_legal_ids": True,
        "seed_files": {"dev_eval": "configs/seeds/dev.txt"},
        "periodic_dev_eval_interval_updates": 10,
        "periodic_dev_eval_paired_seeds": 8,
        "final_policy_set_size": 4,
        "final_matrix_stage1_paired_seeds": 16,
        "final_matrix_stage2_adaptive_max_paired_seeds": 32,
        "stop_rules": {"stop_delta_ci_half_width": 0.05, "stop_confidence": 0.95},
        "replay_capture_rate_eval": 0.1,
        "regression_capture_count": 2,
        "legal_fingerprint_checks": {
            "enabled": True,
            "version": "legal_fingerprint_v1",
            "require_strictly_increasing_legal_ids": True,
            "mismatch_policy": "hard_fail",
        },
        "decision_kind_tagging": {
            "required_for_training": True,
            "enable_python_derived_debug_tag": False,
        },
        "final_policy_set_selection": {
            "version": "final_policy_set_v1",
            "include_random_legal_baseline_b0": True,
            "include_no_league_baseline_b1": True,
            "include_heuristic_public_b2_if_exists": True,
            "include_final_champion_snapshot": True,
            "include_spaced_snapshots_near_percent_updates": [25, 50, 75],
            "remaining_slots_strategy": "recent",
            "fixed_anchor_set_v1": {"required": ["b0"], "optional_if_available": ["b1"]},
            "seed_file": "configs/seeds/final.txt",
            "folding": "seat_swap_mean",
            "seat_swap": True,
            "tie_break": "policy_id",
        },
    }


def test_parse_evaluation_config_accepts_policy_selection_contract() -> None:
    config = parse_evaluation_config(_evaluation_body())

    assert config.seat_swap is True
    assert config.seed_files == {"dev_eval": "configs/seeds/dev.txt"}
    assert config.stop_rules.stop_confidence == pytest.approx(0.95)
    assert config.legal_fingerprint_checks.mismatch_policy == "hard_fail"
    assert config.decision_kind_tagging.required_for_training is True
    assert config.model_sampling_temperature == pytest.approx(1.0)
    assert config.final_policy_set_selection.include_spaced_snapshots_near_percent_updates == (25, 50, 75)
    assert config.final_policy_set_selection.fixed_anchor_set_v1.required == ("b0",)
    assert config.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == ("b1",)
    assert config.final_policy_set_selection.seed_file == "configs/seeds/final.txt"


def test_parse_evaluation_config_preserves_hard_fail_guard() -> None:
    body = _evaluation_body()
    legal = _copy_section(body, "legal_fingerprint_checks")
    legal["mismatch_policy"] = "ignore"
    body["legal_fingerprint_checks"] = legal

    with pytest.raises(
        ValueError,
        match="evaluation.legal_fingerprint_checks.mismatch_policy must be 'hard_fail', got 'ignore'",
    ):
        parse_evaluation_config(body)


def test_parse_evaluation_config_preserves_unknown_key_errors() -> None:
    unknown = _evaluation_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="evaluation has unsupported keys: extra"):
        parse_evaluation_config(unknown)

    bad_selection = _evaluation_body()
    selection = _copy_section(bad_selection, "final_policy_set_selection")
    selection["extra"] = True
    bad_selection["final_policy_set_selection"] = selection
    with pytest.raises(ValueError, match="evaluation.final_policy_set_selection has unsupported keys: extra"):
        parse_evaluation_config(bad_selection)

    bad_anchor = _evaluation_body()
    selection = _copy_section(bad_anchor, "final_policy_set_selection")
    fixed_anchor = dict(cast(Mapping[str, object], selection["fixed_anchor_set_v1"]))
    fixed_anchor["extra"] = True
    selection["fixed_anchor_set_v1"] = fixed_anchor
    bad_anchor["final_policy_set_selection"] = selection
    with pytest.raises(
        ValueError,
        match="evaluation.final_policy_set_selection.fixed_anchor_set_v1 has unsupported keys: extra",
    ):
        parse_evaluation_config(bad_anchor)


def test_parse_evaluation_config_preserves_type_and_minimum_errors() -> None:
    bad_seed = _evaluation_body()
    seed_files = _copy_section(bad_seed, "seed_files")
    seed_files["dev_eval"] = ""
    bad_seed["seed_files"] = seed_files
    with pytest.raises(ValueError, match="evaluation.seed_files.dev_eval must be a non-empty string"):
        parse_evaluation_config(bad_seed)

    bad_periodic = _evaluation_body()
    bad_periodic["periodic_dev_eval_interval_updates"] = -1
    with pytest.raises(ValueError, match="evaluation.periodic_dev_eval_interval_updates must be >= 0, got -1"):
        parse_evaluation_config(bad_periodic)

    bad_selection = _evaluation_body()
    selection = _copy_section(bad_selection, "final_policy_set_selection")
    selection["include_spaced_snapshots_near_percent_updates"] = [25, True]
    bad_selection["final_policy_set_selection"] = selection
    with pytest.raises(
        ValueError,
        match=(
            r"evaluation.final_policy_set_selection.include_spaced_snapshots_near_percent_updates\[\] "
            "must be an integer, got bool"
        ),
    ):
        parse_evaluation_config(bad_selection)

    bad_temperature = _evaluation_body()
    bad_temperature["model_sampling_temperature"] = 0.0
    with pytest.raises(ValueError, match="evaluation.model_sampling_temperature must be > 0"):
        parse_evaluation_config(bad_temperature)


def test_parse_evaluation_config_accepts_model_sampling_temperature() -> None:
    body = _evaluation_body()
    body["model_sampling_temperature"] = 0.25

    config = parse_evaluation_config(body)

    assert config.model_sampling_temperature == pytest.approx(0.25)
