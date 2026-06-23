from __future__ import annotations

import torch
from weiss_rl.models.feature_gathering import (
    gather_stage_features,
    gather_stage_features_for_rows,
    slot_component,
)


def test_gather_stage_features_for_rows_selects_valid_row_slot_pairs_and_zeroes_invalid() -> None:
    slot_contexts = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    slot_numeric = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

    gathered_context, gathered_numeric = gather_stage_features_for_rows(
        slot_contexts,
        slot_numeric,
        torch.tensor([0, 1, 1, 0], dtype=torch.long),
        torch.tensor([2, 1, -1, 3], dtype=torch.long),
        stage_slot_count=3,
    )

    assert torch.equal(
        gathered_context,
        torch.stack(
            [
                slot_contexts[0, 2],
                slot_contexts[1, 1],
                torch.zeros(2),
                torch.zeros(2),
            ],
        ),
    )
    assert torch.equal(
        gathered_numeric,
        torch.stack(
            [
                slot_numeric[0, 2],
                slot_numeric[1, 1],
                torch.zeros(3),
                torch.zeros(3),
            ],
        ),
    )


def test_gather_stage_features_handles_batched_3d_inputs() -> None:
    slot_contexts = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2)
    slot_numeric = torch.arange(3 * 3 * 3, dtype=torch.float32).reshape(3, 3, 3)

    gathered_context, gathered_numeric = gather_stage_features(
        slot_contexts,
        slot_numeric,
        torch.tensor([2, -1, 1], dtype=torch.long),
        stage_slot_count=3,
    )

    assert torch.equal(
        gathered_context,
        torch.stack(
            [
                slot_contexts[0, 2],
                torch.zeros(2),
                slot_contexts[2, 1],
            ],
        ),
    )
    assert torch.equal(
        gathered_numeric,
        torch.stack(
            [
                slot_numeric[0, 2],
                torch.zeros(3),
                slot_numeric[2, 1],
            ],
        ),
    )


def test_gather_stage_features_handles_shared_2d_slot_tables() -> None:
    slot_contexts = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    slot_numeric = torch.arange(3 * 3, dtype=torch.float32).reshape(3, 3)

    gathered_context, gathered_numeric = gather_stage_features(
        slot_contexts,
        slot_numeric,
        torch.tensor([2, -1, 1], dtype=torch.long),
        stage_slot_count=3,
    )

    assert torch.equal(
        gathered_context,
        torch.stack(
            [
                slot_contexts[2],
                torch.zeros(2),
                slot_contexts[1],
            ],
        ),
    )
    assert torch.equal(
        gathered_numeric,
        torch.stack(
            [
                slot_numeric[2],
                torch.zeros(3),
                slot_numeric[1],
            ],
        ),
    )


def test_slot_component_returns_selected_or_missing_component_plane() -> None:
    stage_values = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

    assert torch.equal(slot_component(stage_values, 1), stage_values[..., 1])
    assert torch.equal(slot_component(stage_values, 5), torch.zeros((2, 3)))
