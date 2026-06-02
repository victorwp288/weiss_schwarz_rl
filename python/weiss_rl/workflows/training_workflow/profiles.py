from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainProfile:
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
