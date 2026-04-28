"""Small distributed-training primitives for local-testable learner scaling."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True, slots=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "gloo"
    initialized: bool = False

    @property
    def is_rank0(self) -> bool:
        return int(self.rank) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "world_size": int(self.world_size),
            "backend": str(self.backend),
            "initialized": bool(self.initialized),
        }


def distributed_context_from_env(*, force: bool = False, backend: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)) or "0")
    enabled = bool(force or world_size > 1)
    if not enabled:
        return DistributedContext(enabled=False)
    selected_backend = str(backend).strip().lower()
    if selected_backend == "auto":
        selected_backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=max(1, world_size),
        backend=selected_backend,
        initialized=dist.is_available() and dist.is_initialized(),
    )


def init_process_group_if_needed(context: DistributedContext, *, timeout_seconds: int = 120) -> DistributedContext:
    if not context.enabled:
        return context
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if dist.is_initialized():
        return DistributedContext(
            enabled=True,
            rank=dist.get_rank(),
            local_rank=int(context.local_rank),
            world_size=dist.get_world_size(),
            backend=str(context.backend),
            initialized=True,
        )
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = _free_loopback_port()
    dist.init_process_group(
        backend=str(context.backend),
        rank=int(context.rank),
        world_size=int(context.world_size),
        timeout=timedelta(seconds=int(timeout_seconds)),
    )
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=int(context.local_rank),
        world_size=dist.get_world_size(),
        backend=str(context.backend),
        initialized=True,
    )


def destroy_process_group_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, *, context: DistributedContext, src: int = 0) -> Any:
    if not context.enabled:
        return value
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed object broadcast requires an initialized process group")
    payload = [value if int(context.rank) == int(src) else None]
    dist.broadcast_object_list(payload, src=int(src))
    return payload[0]


def average_gradients(model: nn.Module, *, context: DistributedContext) -> None:
    """Average local gradients across ranks without wrapping the model in DDP.

    The learner keeps the raw model interface because the structured policy path
    calls custom methods that DDP does not transparently expose.
    """

    if not context.enabled:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("gradient averaging requires an initialized process group")
    world_size = int(context.world_size)
    if world_size <= 1:
        return
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(world_size))


def all_reduce_float(value: float, *, context: DistributedContext, op: str = "sum") -> float:
    if not context.enabled:
        return float(value)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("float all-reduce requires an initialized process group")
    tensor = torch.tensor(float(value), dtype=torch.float64)
    reduce_op = dist.ReduceOp.SUM
    dist.all_reduce(tensor, op=reduce_op)
    reduced = float(tensor.item())
    if str(op).strip().lower() == "mean":
        return reduced / float(max(1, int(context.world_size)))
    if str(op).strip().lower() != "sum":
        raise ValueError("op must be 'sum' or 'mean'")
    return reduced


def shard_env_count(*, global_num_envs: int, world_size: int, rank: int) -> int:
    del rank
    total = int(global_num_envs)
    size = int(world_size)
    if size <= 1:
        return total
    if total % size != 0:
        raise ValueError(f"global num_envs={total} must be divisible by DDP world_size={size}")
    return total // size


def rank_seed(base_seed: int, *, rank: int) -> int:
    return int((int(base_seed) ^ ((int(rank) + 1) * 0x9E37_79B9)) & 0x7FFF_FFFF)


def _free_loopback_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])
