from __future__ import annotations

import torch
from weiss_rl.models.public_heuristics import (
    attack_public_heuristic_raw,
    move_public_heuristic_raw,
    play_public_heuristic_raw,
)


def test_play_public_heuristic_raw_rewards_open_preferred_stage_slots() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    stage_slots = torch.tensor([0, 3, 5], dtype=torch.long)
    target_numeric = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float32,
    )

    raw = play_public_heuristic_raw(stage_slots, target_numeric, preferences, dtype=torch.float32)

    assert torch.equal(raw, torch.tensor([710.0, -1000.0, 650.0]))


def test_move_public_heuristic_raw_preserves_slot_improvement_and_validity_rules() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    from_slots = torch.tensor([3, 0, 2], dtype=torch.long)
    to_slots = torch.tensor([1, 4, 1], dtype=torch.long)
    source_numeric = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
    target_numeric = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)

    raw = move_public_heuristic_raw(
        from_slots,
        to_slots,
        source_numeric,
        target_numeric,
        preferences,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([187.0, -1000.0, -1000.0]))


def test_attack_public_heuristic_raw_preserves_attack_type_power_and_occupancy_rules() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    slot_values = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    attack_types = torch.tensor([2, 0, 1, 1], dtype=torch.long)
    source_numeric = torch.zeros((4, 7), dtype=torch.float32)
    source_numeric[:, 0] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    source_numeric[:, 3] = torch.tensor([1.0, 3.0, 2.0, 5.0])
    source_numeric[:, 5] = torch.tensor([2.0, 1.0, 1.0, 1.0])
    source_numeric[:, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    defender_numeric = torch.zeros((4, 7), dtype=torch.float32)
    defender_numeric[:, 0] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    defender_numeric[:, 3] = torch.tensor([0.0, 2.0, 5.0, 1.0])

    raw = attack_public_heuristic_raw(
        slot_values,
        attack_types,
        source_numeric,
        defender_numeric,
        preferences,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([1032.0, 1051.0, 1011.0, -1000.0]))


def test_attack_public_heuristic_raw_uses_catalog_attack_type_ids() -> None:
    preferences = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    slot_values = torch.tensor([0, 1, 2], dtype=torch.long)
    attack_types = torch.tensor([11, 7, 5], dtype=torch.long)
    source_numeric = torch.zeros((3, 7), dtype=torch.float32)
    source_numeric[:, 0] = 1.0
    source_numeric[:, 3] = torch.tensor([1.0, 3.0, 2.0])
    source_numeric[:, 5] = 0.0
    source_numeric[:, 6] = torch.tensor([0.0, 0.0, 1.0])
    defender_numeric = torch.zeros((3, 7), dtype=torch.float32)
    defender_numeric[:, 0] = torch.tensor([0.0, 1.0, 1.0])
    defender_numeric[:, 3] = torch.tensor([0.0, 2.0, 5.0])

    raw = attack_public_heuristic_raw(
        slot_values,
        attack_types,
        source_numeric,
        defender_numeric,
        preferences,
        direct_attack_type_id=11,
        frontal_attack_type_id=7,
        side_attack_type_id=5,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([980.0, 1005.0, 980.0]))
