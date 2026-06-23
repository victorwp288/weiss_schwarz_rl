from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    dense_group_log_probs,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
    structured_group_lookup,
)

from .structured_auxiliary_test_support import _catalog


def test_structured_catalog_metadata_records_family_slots_attacks_and_main_move_pressure() -> None:
    metadata = structured_catalog_metadata(_catalog())

    assert metadata.family_names == ("main_play_character", "attack", "main_move", "pass")
    assert metadata.attack_type_names == ("frontal", "direct", "side")
    assert metadata.family_ids[0] == 0
    assert metadata.family_ids[10] == 1
    assert metadata.family_ids[13] == 2
    assert metadata.play_slots[0] == 0
    assert metadata.play_slots[8] == 3
    assert metadata.attack_slots[10] == 0
    assert metadata.attack_types[11] == 1
    assert metadata.move_from_slots[14] == 0
    assert metadata.move_to_slots[14] == 2
    assert metadata.main_move_02_action_id == 14


def test_structured_group_lookup_builds_dense_action_tables_on_requested_device() -> None:
    catalog = _catalog()

    lookup = structured_group_lookup(catalog, device=torch.device("cpu"))

    assert lookup["family_names"] == ("main_play_character", "attack", "main_move", "pass")
    assert lookup["family_index"] == {"main_play_character": 0, "attack": 1, "main_move": 2, "pass": 3}
    assert lookup["attack_type_names"] == ("frontal", "direct", "side")
    assert lookup["family_ids"].device.type == "cpu"
    assert lookup["family_ids"].tolist()[0] == 0
    assert lookup["play_slots"].tolist()[8] == 3
    assert lookup["move_to_slots"].tolist()[14] == 2
    assert lookup["attack_types"].tolist()[11] == 1


def test_dense_group_log_probs_matches_manual_group_logsumexp() -> None:
    masked_logits = torch.tensor([[2.0, 0.0, -1.0, 4.0], [1.0, 3.0, 5.0, -2.0]], dtype=torch.float32)
    group_ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    log_probs = dense_group_log_probs(masked_logits=masked_logits, group_ids=group_ids, group_count=4)

    expected_rows: list[torch.Tensor] = []
    for row in masked_logits:
        row_log_z = torch.logsumexp(row, dim=0)
        expected_rows.append(
            torch.stack(
                [
                    row[0] - row_log_z,
                    torch.logsumexp(row[1:3], dim=0) - row_log_z,
                    row[3] - row_log_z,
                    torch.tensor(-1.0e9, dtype=row.dtype) - row_log_z,
                ]
            )
        )
    torch.testing.assert_close(log_probs, torch.stack(expected_rows, dim=0))


def test_resolve_public_heuristic_family_ids_preserves_order_and_reports_unknowns() -> None:
    family_names = ("main_play_character", "attack", "main_move", "pass")

    assert resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=(" attack ", "main_move"),
    ) == (1, 2)
    assert resolve_public_heuristic_family_ids(family_names=family_names, requested_families=("", " ")) == ()

    with pytest.raises(ValueError, match="unknown action families: climax"):
        resolve_public_heuristic_family_ids(family_names=family_names, requested_families=("climax",))
