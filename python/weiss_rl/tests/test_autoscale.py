from __future__ import annotations

import pytest

import weiss_rl.autoscale as autoscale
from weiss_rl.autoscale import (
    ScalingRequest,
    detect_hardware_profile,
    hardware_profile_from_name,
    resolve_training_topology,
    validate_ddp_world_size,
)


def _resolve(profile: str):
    return resolve_training_topology(
        hardware=hardware_profile_from_name(profile),
        request=ScalingRequest(
            learner_parallelism="auto",
            learner_gpu_count="auto",
            actor_topology="auto",
            target_envs_per_gpu=512,
            min_envs_per_actor=32,
            max_envs_per_actor=64,
            max_actor_process_count=64,
        ),
        configured_actor_count=12,
        configured_envs_per_actor=8,
        configured_batch_unrolls_per_update=128,
        configured_queue_capacity_unrolls=256,
        runtime_mode="train_async_fast",
    )


def test_l40_3_profile_resolves_server_sized_actor_topology() -> None:
    topology = _resolve("uc1-l40-3")

    assert topology.resolved_learner_parallelism == "ddp"
    assert topology.learner_gpu_count == 3
    assert topology.total_envs == 1536
    assert topology.actor_count == 24
    assert topology.envs_per_actor == 64
    assert topology.batch_unrolls_per_update % topology.learner_gpu_count == 0
    assert topology.queue_capacity_unrolls == 258
    assert topology.queue_capacity_unrolls % topology.learner_gpu_count == 0


def test_l40_4_profile_resolves_server_sized_actor_topology() -> None:
    topology = _resolve("uc1-l40-4")

    assert topology.resolved_learner_parallelism == "ddp"
    assert topology.learner_gpu_count == 4
    assert topology.total_envs == 2048
    assert topology.actor_count == 32
    assert topology.envs_per_actor == 64


def test_generic_gpu8_profile_caps_to_64_actor_processes() -> None:
    topology = _resolve("gpu8")

    assert topology.learner_gpu_count == 8
    assert topology.total_envs == 4096
    assert topology.actor_count == 64
    assert topology.envs_per_actor == 64


def test_validate_ddp_world_size_requires_one_rank_per_learner_gpu() -> None:
    topology = _resolve("uc1-l40-4")

    validate_ddp_world_size(topology, world_size=4)
    with pytest.raises(ValueError, match="torchrun --nproc_per_node=4"):
        validate_ddp_world_size(topology, world_size=2)


def test_explicit_learner_gpu_count_cannot_exceed_visible_gpus() -> None:
    with pytest.raises(ValueError, match="exceeds visible CUDA devices"):
        resolve_training_topology(
            hardware=hardware_profile_from_name("gpu3"),
            request=ScalingRequest(learner_gpu_count="4"),
            configured_actor_count=32,
            configured_envs_per_actor=8,
            configured_batch_unrolls_per_update=64,
            configured_queue_capacity_unrolls=128,
            runtime_mode="train_async_fast",
        )


def test_autoscale_actor_process_cap_is_a_hard_ceiling() -> None:
    topology = resolve_training_topology(
        hardware=hardware_profile_from_name("gpu4"),
        request=ScalingRequest(max_actor_process_count=8),
        configured_actor_count=32,
        configured_envs_per_actor=8,
        configured_batch_unrolls_per_update=64,
        configured_queue_capacity_unrolls=128,
        runtime_mode="train_async_fast",
    )

    assert topology.actor_count <= 8
    assert topology.envs_per_actor <= 64
    assert "target_envs_reduced_by_actor_capacity" in topology.notes


def test_autoscale_fails_when_actor_capacity_cannot_cover_ddp_ranks() -> None:
    with pytest.raises(ValueError, match="one actor per DDP rank"):
        resolve_training_topology(
            hardware=hardware_profile_from_name("gpu4"),
            request=ScalingRequest(max_actor_process_count=2),
            configured_actor_count=32,
            configured_envs_per_actor=8,
            configured_batch_unrolls_per_update=64,
            configured_queue_capacity_unrolls=128,
            runtime_mode="train_async_fast",
        )


def test_manual_actor_topology_must_be_shardable_for_ddp() -> None:
    with pytest.raises(ValueError, match="manual actor topology"):
        resolve_training_topology(
            hardware=hardware_profile_from_name("gpu2"),
            request=ScalingRequest(actor_topology="manual"),
            configured_actor_count=3,
            configured_envs_per_actor=64,
            configured_batch_unrolls_per_update=64,
            configured_queue_capacity_unrolls=128,
            runtime_mode="train_async_fast",
        )


def test_manual_actor_topology_obeys_actor_process_cap() -> None:
    with pytest.raises(ValueError, match="exceeds max_actor_process_count"):
        resolve_training_topology(
            hardware=hardware_profile_from_name("gpu4"),
            request=ScalingRequest(actor_topology="manual", max_actor_process_count=64),
            configured_actor_count=96,
            configured_envs_per_actor=64,
            configured_batch_unrolls_per_update=128,
            configured_queue_capacity_unrolls=256,
            runtime_mode="train_async_fast",
        )


def test_detect_hardware_profile_uses_smallest_visible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    class Props:
        def __init__(self, name: str, total_memory: int) -> None:
            self.name = name
            self.total_memory = total_memory

    gib = 1024**3
    devices = [
        Props("NVIDIA A100", 80 * gib),
        Props("NVIDIA L40", 48 * gib),
        Props("NVIDIA L40", 48 * gib),
    ]

    monkeypatch.setattr(autoscale.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(autoscale.torch.cuda, "device_count", lambda: len(devices))
    monkeypatch.setattr(autoscale.torch.cuda, "get_device_properties", lambda index: devices[index])

    profile = detect_hardware_profile()

    assert profile.gpu_count == 3
    assert profile.vram_gb_per_gpu == 48.0
    assert profile.gpu_name == "mixed: NVIDIA A100, NVIDIA L40"
