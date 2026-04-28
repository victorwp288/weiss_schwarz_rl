from __future__ import annotations

import pytest

from weiss_rl.distributed import DistributedContext, all_reduce_float, rank_seed, shard_env_count


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
