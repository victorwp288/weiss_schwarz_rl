from __future__ import annotations

import torch
from weiss_rl.models.actions.candidate_components import (
    CandidateComponentFamilyIds,
    resolve_candidate_components,
)


def _family_ids() -> CandidateComponentFamilyIds:
    return CandidateComponentFamilyIds(
        play_character=10,
        main_event=11,
        clock_from_hand=12,
        climax_play=13,
        mulligan_select=14,
        main_move=20,
        attack=30,
        choice_select=40,
        level_up=41,
        trigger_order=42,
    )


def test_resolve_candidate_components_uses_action_id_tables_without_meta() -> None:
    result = resolve_candidate_components(
        torch.tensor([2, 0], dtype=torch.long),
        None,
        family_ids_by_action=torch.tensor([10, 20, 30], dtype=torch.long),
        hand_indices_by_action=torch.tensor([1, -1, -1], dtype=torch.long),
        stage_slots_by_action=torch.tensor([2, -1, -1], dtype=torch.long),
        from_slots_by_action=torch.tensor([-1, 3, -1], dtype=torch.long),
        to_slots_by_action=torch.tensor([-1, 4, -1], dtype=torch.long),
        attack_slots_by_action=torch.tensor([-1, -1, 0], dtype=torch.long),
        attack_types_by_action=torch.tensor([-1, -1, 2], dtype=torch.long),
        generic_indices_by_action=torch.tensor([-1, -1, -1], dtype=torch.long),
        meta_unused=65535,
        family_ids=_family_ids(),
    )

    assert [tensor.tolist() for tensor in result] == [
        [30, 10],
        [-1, 1],
        [-1, 2],
        [-1, -1],
        [-1, -1],
        [0, -1],
        [2, -1],
        [-1, -1],
    ]


def test_resolve_candidate_components_decodes_packed_meta_by_family() -> None:
    unused = 65535
    meta = torch.tensor(
        [
            [10, 1, 2],
            [20, 3, 4],
            [30, 0, 2],
            [40, 7, unused],
            [11, 5, unused],
            [99, unused, unused],
        ],
        dtype=torch.long,
    )

    result = resolve_candidate_components(
        torch.arange(meta.shape[0], dtype=torch.long),
        meta,
        family_ids_by_action=torch.empty((0,), dtype=torch.long),
        hand_indices_by_action=torch.empty((0,), dtype=torch.long),
        stage_slots_by_action=torch.empty((0,), dtype=torch.long),
        from_slots_by_action=torch.empty((0,), dtype=torch.long),
        to_slots_by_action=torch.empty((0,), dtype=torch.long),
        attack_slots_by_action=torch.empty((0,), dtype=torch.long),
        attack_types_by_action=torch.empty((0,), dtype=torch.long),
        generic_indices_by_action=torch.empty((0,), dtype=torch.long),
        meta_unused=unused,
        family_ids=_family_ids(),
    )

    assert [tensor.tolist() for tensor in result] == [
        [10, 20, 30, 40, 11, 99],
        [1, -1, -1, -1, 5, -1],
        [2, -1, -1, -1, -1, -1],
        [-1, 3, -1, -1, -1, -1],
        [-1, 4, -1, -1, -1, -1],
        [-1, -1, 0, -1, -1, -1],
        [-1, -1, 2, -1, -1, -1],
        [-1, -1, -1, 7, -1, -1],
    ]


def test_resolve_candidate_components_ignores_missing_family_ids() -> None:
    families = CandidateComponentFamilyIds(
        play_character=-1,
        main_event=-1,
        clock_from_hand=-1,
        climax_play=-1,
        mulligan_select=-1,
        main_move=-1,
        attack=-1,
        choice_select=-1,
        level_up=-1,
        trigger_order=-1,
    )

    result = resolve_candidate_components(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[10, 1, 2]], dtype=torch.long),
        family_ids_by_action=torch.empty((0,), dtype=torch.long),
        hand_indices_by_action=torch.empty((0,), dtype=torch.long),
        stage_slots_by_action=torch.empty((0,), dtype=torch.long),
        from_slots_by_action=torch.empty((0,), dtype=torch.long),
        to_slots_by_action=torch.empty((0,), dtype=torch.long),
        attack_slots_by_action=torch.empty((0,), dtype=torch.long),
        attack_types_by_action=torch.empty((0,), dtype=torch.long),
        generic_indices_by_action=torch.empty((0,), dtype=torch.long),
        meta_unused=65535,
        family_ids=families,
    )

    assert [tensor.tolist() for tensor in result] == [[10], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
