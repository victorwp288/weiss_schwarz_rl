"""Hardware-aware training topology resolution.

The resolver is intentionally deterministic: live hardware detection is only
used to produce an input profile, and the resolved topology is written into
artifacts so an ``auto`` run can be reproduced or audited later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    name: str
    cpu_cores: int
    ram_gb: float
    gpu_count: int
    vram_gb_per_gpu: float
    gpu_name: str = ""


@dataclass(frozen=True, slots=True)
class ScalingRequest:
    learner_parallelism: str = "auto"
    learner_gpu_count: str = "auto"
    actor_topology: str = "auto"
    target_envs_per_gpu: int = 512
    min_envs_per_actor: int = 32
    max_envs_per_actor: int = 64
    max_actor_process_count: int = 64
    reserve_cpu_cores: int = 4
    learner_cpu_cores_per_gpu: int = 2
    queue_depth_multiplier: int = 2
    ram_queue_fraction: float = 0.25
    vram_fraction: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingTopology:
    requested_learner_parallelism: str
    resolved_learner_parallelism: str
    requested_learner_gpu_count: str
    learner_gpu_count: int
    visible_gpu_count: int
    gpu_name: str
    cpu_cores: int
    ram_gb: float
    vram_gb_per_gpu: float
    actor_count: int
    envs_per_actor: int
    total_envs: int
    batch_unrolls_per_update: int
    queue_capacity_unrolls: int
    learner_reserved_cpu_cores: int
    actor_cpu_cores_available: int
    ram_queue_budget_gb: float
    vram_budget_gb_per_gpu: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MOCK_PROFILES: dict[str, HardwareProfile] = {
    "uc1-l40-3": HardwareProfile(
        name="uc1-l40-3",
        cpu_cores=72,
        ram_gb=288.0,
        gpu_count=3,
        vram_gb_per_gpu=48.0,
        gpu_name="NVIDIA L40",
    ),
    "uc1-l40-4": HardwareProfile(
        name="uc1-l40-4",
        cpu_cores=96,
        ram_gb=384.0,
        gpu_count=4,
        vram_gb_per_gpu=48.0,
        gpu_name="NVIDIA L40",
    ),
    "8gpu-l40": HardwareProfile(
        name="8gpu-l40",
        cpu_cores=192,
        ram_gb=768.0,
        gpu_count=8,
        vram_gb_per_gpu=48.0,
        gpu_name="NVIDIA L40",
    ),
}


def hardware_profile_from_name(name: str) -> HardwareProfile:
    normalized = str(name).strip().lower()
    if normalized in {"", "auto", "local"}:
        return detect_hardware_profile(name="local")
    if normalized in _MOCK_PROFILES:
        return _MOCK_PROFILES[normalized]
    if normalized.startswith("gpu") and normalized[3:].isdigit():
        gpu_count = int(normalized[3:])
        return HardwareProfile(
            name=normalized,
            cpu_cores=max(16, gpu_count * 24),
            ram_gb=float(max(64, gpu_count * 96)),
            gpu_count=gpu_count,
            vram_gb_per_gpu=48.0,
            gpu_name="mock-gpu",
        )
    raise ValueError(
        "unknown hardware profile "
        f"{name!r}; use local, uc1-l40-3, uc1-l40-4, 8gpu-l40, or gpu<N>"
    )


def detect_hardware_profile(*, name: str = "local") -> HardwareProfile:
    cpu_cores = int(psutil.cpu_count(logical=True) or 1)
    ram_gb = float(psutil.virtual_memory().total / (1024.0**3))
    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    gpu_name = ""
    vram_gb = 0.0
    if gpu_count > 0:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            vram_gb = float(props.total_memory / (1024.0**3))
        except Exception:
            gpu_name = "cuda"
            vram_gb = 0.0
    return HardwareProfile(
        name=str(name),
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_count=gpu_count,
        vram_gb_per_gpu=vram_gb,
        gpu_name=gpu_name,
    )


def _parse_gpu_count(requested: str, *, visible_gpu_count: int) -> int:
    text = str(requested).strip().lower()
    if text == "auto":
        return max(1, int(visible_gpu_count)) if visible_gpu_count > 0 else 0
    gpu_count = int(text)
    if gpu_count < 0:
        raise ValueError("learner_gpu_count must be >= 0 or auto")
    if visible_gpu_count > 0:
        return min(gpu_count, visible_gpu_count)
    return gpu_count


def resolve_training_topology(
    *,
    hardware: HardwareProfile,
    request: ScalingRequest,
    configured_actor_count: int,
    configured_envs_per_actor: int,
    configured_batch_unrolls_per_update: int,
    configured_queue_capacity_unrolls: int,
    runtime_mode: str,
) -> ResolvedTrainingTopology:
    notes: list[str] = []
    visible_gpu_count = max(0, int(hardware.gpu_count))
    learner_gpu_count = _parse_gpu_count(request.learner_gpu_count, visible_gpu_count=visible_gpu_count)
    requested_parallelism = str(request.learner_parallelism).strip().lower()
    if requested_parallelism == "auto":
        resolved_parallelism = "ddp" if learner_gpu_count > 1 else ("single_cuda" if learner_gpu_count == 1 else "single_cpu")
    elif requested_parallelism == "single":
        resolved_parallelism = "single_cuda" if learner_gpu_count > 0 else "single_cpu"
        learner_gpu_count = min(learner_gpu_count, 1)
    elif requested_parallelism in {"ddp", "ddp_cpu_test"}:
        resolved_parallelism = requested_parallelism
    else:
        raise ValueError("learner_parallelism must be one of: auto, single, ddp, ddp_cpu_test")

    if str(request.actor_topology).strip().lower() != "auto":
        actor_count = max(1, int(configured_actor_count))
        envs_per_actor = max(1, int(configured_envs_per_actor))
        notes.append("manual_actor_topology")
    else:
        scale_gpu_count = max(1, int(learner_gpu_count))
        target_total_envs = max(1, int(request.target_envs_per_gpu) * scale_gpu_count)
        learner_reserved = max(1, int(request.learner_cpu_cores_per_gpu) * max(1, int(learner_gpu_count)))
        actor_cpu_budget = max(1, int(hardware.cpu_cores) - int(request.reserve_cpu_cores) - learner_reserved)
        requested_actor_cap = max(int(configured_actor_count), int(request.max_actor_process_count))
        max_actor_count = max(1, min(requested_actor_cap, actor_cpu_budget))
        min_envs = max(1, int(request.min_envs_per_actor))
        max_envs = max(min_envs, int(request.max_envs_per_actor))
        actor_candidates = [
            actor_count
            for actor_count in range(1, max_actor_count + 1)
            if target_total_envs % actor_count == 0
            and min_envs <= (target_total_envs // actor_count) <= max_envs
        ]
        if not actor_candidates:
            actor_count = max(1, min(max_actor_count, max(1, target_total_envs // max_envs)))
            envs_per_actor = max(1, target_total_envs // actor_count)
            notes.append("actor_topology_rounded")
        else:
            actor_count = min(
                actor_candidates,
                key=lambda candidate: (abs(max_envs - (target_total_envs // candidate)), candidate),
            )
            envs_per_actor = target_total_envs // actor_count
    total_envs = int(actor_count * envs_per_actor)
    batch_unrolls_per_update = max(int(configured_batch_unrolls_per_update), int(actor_count))
    queue_capacity_unrolls = max(
        int(configured_queue_capacity_unrolls),
        int(batch_unrolls_per_update) * max(1, int(request.queue_depth_multiplier)),
    )
    learner_reserved_cpu = max(1, int(request.learner_cpu_cores_per_gpu) * max(1, int(learner_gpu_count)))
    actor_cpu_available = max(1, int(hardware.cpu_cores) - int(request.reserve_cpu_cores) - learner_reserved_cpu)
    return ResolvedTrainingTopology(
        requested_learner_parallelism=requested_parallelism,
        resolved_learner_parallelism=resolved_parallelism,
        requested_learner_gpu_count=str(request.learner_gpu_count),
        learner_gpu_count=int(learner_gpu_count),
        visible_gpu_count=int(visible_gpu_count),
        gpu_name=str(hardware.gpu_name),
        cpu_cores=int(hardware.cpu_cores),
        ram_gb=float(hardware.ram_gb),
        vram_gb_per_gpu=float(hardware.vram_gb_per_gpu),
        actor_count=int(actor_count),
        envs_per_actor=int(envs_per_actor),
        total_envs=int(total_envs),
        batch_unrolls_per_update=int(batch_unrolls_per_update),
        queue_capacity_unrolls=int(queue_capacity_unrolls),
        learner_reserved_cpu_cores=int(learner_reserved_cpu),
        actor_cpu_cores_available=int(actor_cpu_available),
        ram_queue_budget_gb=float(hardware.ram_gb) * float(request.ram_queue_fraction),
        vram_budget_gb_per_gpu=float(hardware.vram_gb_per_gpu) * float(request.vram_fraction),
        notes=tuple(notes),
    )
