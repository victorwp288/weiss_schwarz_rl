from __future__ import annotations

import pytest
import torch

import weiss_rl.distributed as distributed
from weiss_rl.distributed import (
    DistributedContext,
    all_reduce_float,
    average_gradients,
    rank_seed,
    resolve_distributed_learner_device,
    shard_env_count,
)


def test_shard_env_count_requires_even_ddp_shards() -> None:
    assert shard_env_count(global_num_envs=2048, world_size=4, rank=0) == 512
    assert shard_env_count(global_num_envs=2048, world_size=4, rank=3) == 512

    with pytest.raises(ValueError, match="must be divisible"):
        shard_env_count(global_num_envs=2050, world_size=4, rank=0)


def test_rank_seed_is_rank_unique_and_stable() -> None:
    seeds = [rank_seed(12345, rank=rank) for rank in range(8)]

    assert len(set(seeds)) == 8
    assert seeds == [rank_seed(12345, rank=rank) for rank in range(8)]


def test_all_reduce_float_is_noop_without_distributed_context() -> None:
    assert all_reduce_float(3.5, context=DistributedContext(enabled=False), op="sum") == 3.5


def test_average_gradients_preserves_none_when_all_ranks_missing_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.full_like(parameters[0], 4.0)
    parameters[1].grad = None
    parameters[2].requires_grad_(False)
    parameters[2].grad = None

    reduced_shapes: list[tuple[int, ...]] = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        if tensor.shape:
            reduced_shapes.append(tuple(tensor.shape))

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)

    average_gradients(model, context=DistributedContext(enabled=True, world_size=2, initialized=True))

    assert reduced_shapes == [tuple(parameters[0].shape)]
    assert torch.all(parameters[0].grad == 2.0)
    assert parameters[1].grad is None
    assert parameters[2].grad is None
    assert parameters[3].grad is None


def test_average_gradients_materializes_zero_when_another_rank_has_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 1)
    parameters = list(model.parameters())
    parameters[0].grad = None
    parameters[1].grad = torch.full_like(parameters[1], 6.0)

    reduced_shapes: list[tuple[int, ...]] = []
    flag_index = 0
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        nonlocal flag_index
        del op
        if not tensor.shape:
            if flag_index == 0:
                tensor.fill_(1.0)
            flag_index += 1
            return
        reduced_shapes.append(tuple(tensor.shape))

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)

    average_gradients(model, context=DistributedContext(enabled=True, world_size=2, initialized=True))

    assert reduced_shapes == [tuple(parameters[0].shape), tuple(parameters[1].shape)]
    assert parameters[0].grad is not None
    assert torch.all(parameters[0].grad == 0.0)
    assert torch.all(parameters[1].grad == 3.0)


def test_nccl_scalar_all_reduce_uses_local_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed.dist, "is_available", lambda: False)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 2)

    context = DistributedContext(enabled=True, local_rank=3, world_size=4, backend="nccl", initialized=True)

    assert distributed._scalar_all_reduce_device(context) == torch.device("cuda:1")


def test_distributed_device_bare_cuda_uses_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 4)

    context = DistributedContext(enabled=True, local_rank=6, world_size=8, backend="nccl", initialized=True)

    assert resolve_distributed_learner_device("cuda", context=context) == torch.device("cuda:2")
    assert resolve_distributed_learner_device("cuda:auto", context=context) == torch.device("cuda:2")


def test_distributed_device_rejects_mismatched_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 4)
    context = DistributedContext(enabled=True, local_rank=2, world_size=4, backend="nccl", initialized=True)

    with pytest.raises(ValueError, match="does not match this rank"):
        resolve_distributed_learner_device("cuda:0", context=context)
