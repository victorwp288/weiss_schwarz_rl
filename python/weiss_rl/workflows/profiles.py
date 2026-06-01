"""Standard package-CLI workflow profiles and config paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "B1_GUIDED_SEED_STACK_CONFIG",
    "B1_STACK_CONFIG",
    "EVAL_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG",
    "MAIN_STACK_CONFIG",
    "TRAIN_PROFILES",
    "TrainProfile",
    "guided_bootstrap_stack_config",
    "resolve_repo_root",
]


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
B1_GUIDED_SEED_STACK_CONFIG = Path("configs/thesis/b1_guided_seed.yaml")
MAIN_STACK_CONFIG = Path("configs/thesis/main_league.yaml")
MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap.yaml")
MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_vtrace.yaml")
MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_seedchampion.yaml")
MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
)
MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml"
)
EVAL_STACK_CONFIG = Path("configs/thesis/final_eval.yaml")


def resolve_repo_root(args_repo_root: Path | None) -> Path:
    return Path(__file__).resolve().parents[3] if args_repo_root is None else args_repo_root.resolve()


def guided_bootstrap_stack_config(
    *,
    vtrace_clamp: bool,
    seed_champions: bool,
    selected_seed_champion: bool,
) -> Path:
    selected_count = sum(bool(value) for value in (vtrace_clamp, seed_champions, selected_seed_champion))
    if selected_count > 1:
        raise SystemExit("--vtrace-clamp, --seed-champions, and --selected-seed-champion select distinct stacks")
    if selected_seed_champion:
        return MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG
    if seed_champions:
        return MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG
    if vtrace_clamp:
        return MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG
    return MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG
