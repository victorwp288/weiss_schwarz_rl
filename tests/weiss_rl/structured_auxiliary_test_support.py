from __future__ import annotations

import torch
from weiss_rl.core.action_catalog import ActionCatalog


def _catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 20,
                "pass_action_id": 19,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 3},
                    {"name": "main_move", "base": 13, "count": 6},
                    {"name": "pass", "base": 19, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def action_margin_packed_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor([1.0, 2.0, 1.7, 4.0, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([0, 10, 19, 0, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 0, -1, -1],
            [1, 0, 0, -1],
            [3, -1, -1, -1],
            [0, 0, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    return logits, packed_ids, packed_offsets, packed_meta


def same_family_margin_packed_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor([2.0, 1.7, 0.0, 4.0, 3.8, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([10, 11, 19, 13, 14, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [3, -1, -1, -1],
            [2, 0, 1, -1],
            [2, 0, 2, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    return logits, packed_ids, packed_offsets, packed_meta
