from __future__ import annotations

from weiss_rl.autoscale import ScalingRequest, hardware_profile_from_name, resolve_training_topology


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
    assert topology.queue_capacity_unrolls == 256


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
