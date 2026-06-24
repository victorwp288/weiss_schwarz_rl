from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weiss_rl.workflows.training_workflow.stages import training_workflow_stage_payload


@dataclass(frozen=True, slots=True)
class TrainProfile:
    purpose: str
    evidence_level: str
    num_envs: int
    unroll_length: int
    max_updates: int
    runtime_mode: str
    simulator_profile: str
    device: str
    checkpoint_interval_updates: int | None
    overrides: tuple[str, ...] = ()


TRAIN_PROFILES: dict[str, TrainProfile] = {
    "smoke": TrainProfile(
        purpose="fast wiring check for commands, configs, simulator import, and artifact paths",
        evidence_level="plumbing_only",
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cpu",
        checkpoint_interval_updates=1,
        overrides=("system.collection_backend=auto",),
    ),
    "gpu-probe": TrainProfile(
        purpose="short CUDA/runtime probe before longer thesis runs",
        evidence_level="runtime_probe",
        num_envs=32,
        unroll_length=16,
        max_updates=2,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=1,
        overrides=("system.collection_backend=auto", "training.profile_timers=true"),
    ),
    "league-probe": TrainProfile(
        purpose="medium league-path probe with profiling before full training",
        evidence_level="collapse_probe",
        num_envs=288,
        unroll_length=64,
        max_updates=50,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=5,
        overrides=("system.collection_backend=process", "training.profile_timers=true"),
    ),
    "thesis-local": TrainProfile(
        purpose="local thesis-grade training geometry for retained runs",
        evidence_level="retained_training",
        num_envs=288,
        unroll_length=64,
        max_updates=200,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=25,
        overrides=("system.collection_backend=auto",),
    ),
    "thesis-server": TrainProfile(
        purpose="server thesis-grade training geometry for retained runs",
        evidence_level="retained_training",
        num_envs=4096,
        unroll_length=64,
        max_updates=200,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=25,
        overrides=("system.collection_backend=process",),
    ),
}

B1_STACK_CONFIG = Path("configs/thesis/b1_noleague.yaml")
MAIN_STACK_CONFIG = Path("configs/thesis/main_league.yaml")


def training_profile_payload(*, profile_name: str, profile: TrainProfile) -> dict[str, object]:
    """Return the profile fields that make workflow dry-runs explainable."""

    return {
        "profile": profile_name,
        "profile_purpose": profile.purpose,
        "profile_evidence_level": profile.evidence_level,
        "num_envs": profile.num_envs,
        "unroll_length": profile.unroll_length,
        "max_updates": profile.max_updates,
        "runtime_mode": profile.runtime_mode,
        "simulator_profile": profile.simulator_profile,
        "device": profile.device,
        "checkpoint_interval_updates": profile.checkpoint_interval_updates,
        "workflow_stages": training_workflow_stage_payload(),
    }
