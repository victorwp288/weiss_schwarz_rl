from __future__ import annotations

import pytest
from weiss_rl.config.sections.sections_training import parse_training_config
from weiss_rl.config.sections.sections_training_sections import (
    resolve_training_section_mappings,
    training_config_section_map_payload,
)

from tests.weiss_rl.config_training_test_support import copy_section, training_body


def test_parse_training_config_preserves_choice_and_range_errors() -> None:
    bad_algorithm = training_body()
    bad_algorithm["algorithm"] = "new_algo"
    with pytest.raises(ValueError, match="training.algorithm must be one of:"):
        parse_training_config(bad_algorithm)

    bad_fraction = training_body()
    bad_fraction["actor_heuristic_fraction"] = 1.5
    with pytest.raises(
        ValueError,
        match="training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive, got 1.5",
    ):
        parse_training_config(bad_fraction)

    bad_schedule = training_body()
    bad_schedule["actor_heuristic_start_updates"] = 10
    bad_schedule["actor_heuristic_end_updates"] = 5
    with pytest.raises(
        ValueError,
        match="training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates",
    ):
        parse_training_config(bad_schedule)

    bad_diverse_fraction = training_body()
    bad_diverse_fraction["diverse_opponent_batch_fraction"] = -0.1
    with pytest.raises(
        ValueError,
        match="training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive, got -0.1",
    ):
        parse_training_config(bad_diverse_fraction)

    bad_temperature = training_body()
    exploration = copy_section(bad_temperature, "exploration")
    exploration["actor_sampling_temperature"] = 0.0
    bad_temperature["exploration"] = exploration
    with pytest.raises(ValueError, match="actor_sampling_temperature must be > 0"):
        parse_training_config(bad_temperature)


def test_parse_training_config_preserves_nested_unknown_and_minimum_errors() -> None:
    unknown = training_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="training has unsupported keys: extra"):
        parse_training_config(unknown)

    bad_rollout = training_body()
    rollout = copy_section(bad_rollout, "rollout")
    rollout["extra"] = True
    bad_rollout["rollout"] = rollout
    with pytest.raises(ValueError, match="training.rollout has unsupported keys: extra"):
        parse_training_config(bad_rollout)

    bad_unroll = training_body()
    rollout = copy_section(bad_unroll, "rollout")
    rollout["unroll_length"] = 0
    bad_unroll["rollout"] = rollout
    with pytest.raises(ValueError, match="training.rollout.unroll_length must be >= 1, got 0"):
        parse_training_config(bad_unroll)


def test_training_section_mappings_own_nested_training_key_validation() -> None:
    body = training_body()

    sections = resolve_training_section_mappings(body)

    assert sections.rollout == body["rollout"]
    assert sections.optimizer == body["optimizer"]
    assert sections.structured_aux == {}

    bad_body = training_body()
    precision = copy_section(bad_body, "precision")
    precision["extra"] = True
    bad_body["precision"] = precision
    with pytest.raises(ValueError, match="training.precision has unsupported keys: extra"):
        resolve_training_section_mappings(bad_body)


def test_training_config_section_map_names_nested_sections() -> None:
    payload = training_config_section_map_payload()

    assert [section["section"] for section in payload] == [
        "rollout",
        "optimizer",
        "exploration",
        "precision",
        "checkpointing",
        "vtrace",
        "ppo",
        "structured_aux",
        "structured_warmstart",
        "structured_metrics",
        "teacher_aux",
        "action_surface",
    ]
    assert payload[5]["purpose"] == "IMPALA off-policy clipping constants"
    assert payload[-1]["key_count"] > 0
