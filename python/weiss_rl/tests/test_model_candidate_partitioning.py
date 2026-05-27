from __future__ import annotations

import torch

from weiss_rl.models.candidate_partitioning import partition_candidate_family_indices


def test_partition_candidate_family_indices_preserves_group_order_and_candidate_order() -> None:
    partitions = partition_candidate_family_indices(
        torch.tensor([10, 12, 20, 30, 41, 50, 99, 11, 51], dtype=torch.long),
        play_character_family_id=10,
        hand_family_ids=(11, 12),
        main_move_family_id=20,
        attack_family_id=30,
        slot_family_ids=(41,),
        index_family_ids=(50, 51),
    )

    assert [partition.tolist() for partition in partitions] == [
        [0],
        [1, 7],
        [2],
        [3],
        [4],
        [5, 8],
        [6],
    ]


def test_partition_candidate_family_indices_returns_empty_long_tensors_for_missing_groups() -> None:
    family_ids = torch.tensor([99], dtype=torch.long)

    partitions = partition_candidate_family_indices(
        family_ids,
        play_character_family_id=10,
        hand_family_ids=(),
        main_move_family_id=20,
        attack_family_id=30,
        slot_family_ids=(),
        index_family_ids=(),
    )

    assert [partition.tolist() for partition in partitions] == [[], [], [], [], [], [], [0]]
    assert all(partition.dtype == torch.long for partition in partitions)
    assert all(partition.device == family_ids.device for partition in partitions)
