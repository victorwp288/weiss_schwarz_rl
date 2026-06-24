"""Opponent-context helpers for policy/value models."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor


def opponent_context_seed(policy_id: str, *, index: int) -> int:
    digest = hashlib.sha256(f"weiss_rl_opponent_context_v1\0{int(index)}\0{str(policy_id)}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def build_opponent_context_offsets(
    *,
    policy_ids: tuple[str, ...],
    hidden_size: int,
    scale: float,
) -> Tensor:
    offsets = torch.zeros((len(policy_ids) + 1, int(hidden_size)), dtype=torch.float32)
    if not policy_ids or float(scale) <= 0.0:
        return offsets
    for index, policy_id in enumerate(policy_ids, start=1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(opponent_context_seed(policy_id, index=index))
        row = torch.randn((int(hidden_size),), generator=generator, dtype=torch.float32)
        row = row / row.norm().clamp_min(1.0)
        offsets[index] = row * float(scale)
    return offsets


__all__ = ["build_opponent_context_offsets", "opponent_context_seed"]
