"""Compact hyperparameter sweep presets for thesis comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiment_launcher import LaunchJob, LaunchPlan


@dataclass(frozen=True, slots=True)
class SweepCandidate:
    candidate_id: str
    description: str
    overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SweepPreset:
    preset_id: str
    stack_config: str
    candidates: tuple[SweepCandidate, ...]


_PRESETS = {
    "noleague_impala_compact": SweepPreset(
        preset_id="noleague_impala_compact",
        stack_config="configs/presets/baselines/noleague_impala.yaml",
        candidates=(
            SweepCandidate(
                candidate_id="impala_lr1p5e4_ent2e2",
                description="Lower entropy, slightly lower learning rate",
                overrides={
                    "training.optimizer.learning_rate": 1.5e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="impala_lr2e4_ent2e2",
                description="Locked learning rate with reduced entropy",
                overrides={
                    "training.optimizer.learning_rate": 2.0e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="impala_lr2e4_ent3e2",
                description="Frozen thesis-model anchor",
                overrides={
                    "training.optimizer.learning_rate": 2.0e-4,
                    "training.exploration.entropy_coef": 3.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="impala_lr3e4_ent2e2",
                description="Higher learning rate stress point",
                overrides={
                    "training.optimizer.learning_rate": 3.0e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
        ),
    ),
    "norecurrence_compact": SweepPreset(
        preset_id="norecurrence_compact",
        stack_config="configs/presets/baselines/norecurrence_impala.yaml",
        candidates=(
            SweepCandidate(
                candidate_id="ff_lr1p5e4_ent2e2",
                description="Lower entropy, slightly lower learning rate",
                overrides={
                    "training.optimizer.learning_rate": 1.5e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="ff_lr2e4_ent2e2",
                description="Locked learning rate with reduced entropy",
                overrides={
                    "training.optimizer.learning_rate": 2.0e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="ff_lr2e4_ent3e2",
                description="Frozen thesis-model anchor",
                overrides={
                    "training.optimizer.learning_rate": 2.0e-4,
                    "training.exploration.entropy_coef": 3.0e-2,
                },
            ),
            SweepCandidate(
                candidate_id="ff_lr3e4_ent2e2",
                description="Higher learning rate stress point",
                overrides={
                    "training.optimizer.learning_rate": 3.0e-4,
                    "training.exploration.entropy_coef": 2.0e-2,
                },
            ),
        ),
    ),
    "ppo_compact": SweepPreset(
        preset_id="ppo_compact",
        stack_config="configs/presets/baselines/ppo_lite.yaml",
        candidates=(
            SweepCandidate(
                candidate_id="ppo_lr1e4_clip02_ent5e3_ep4",
                description="Conservative PPO-lite baseline",
                overrides={
                    "training.optimizer.learning_rate": 1.0e-4,
                    "training.ppo.clip_epsilon": 0.2,
                    "training.exploration.entropy_coef": 5.0e-3,
                    "training.ppo.epochs": 4,
                },
            ),
            SweepCandidate(
                candidate_id="ppo_lr1p5e4_clip02_ent5e3_ep4",
                description="Default PPO-lite anchor",
                overrides={
                    "training.optimizer.learning_rate": 1.5e-4,
                    "training.ppo.clip_epsilon": 0.2,
                    "training.exploration.entropy_coef": 5.0e-3,
                    "training.ppo.epochs": 4,
                },
            ),
            SweepCandidate(
                candidate_id="ppo_lr1p5e4_clip01_ent5e3_ep4",
                description="Tighter clipping ablation",
                overrides={
                    "training.optimizer.learning_rate": 1.5e-4,
                    "training.ppo.clip_epsilon": 0.1,
                    "training.exploration.entropy_coef": 5.0e-3,
                    "training.ppo.epochs": 4,
                },
            ),
            SweepCandidate(
                candidate_id="ppo_lr1p5e4_clip02_ent1e3_ep2",
                description="Lower entropy and fewer epochs",
                overrides={
                    "training.optimizer.learning_rate": 1.5e-4,
                    "training.ppo.clip_epsilon": 0.2,
                    "training.exploration.entropy_coef": 1.0e-3,
                    "training.ppo.epochs": 2,
                },
            ),
        ),
    ),
}


def list_sweep_presets() -> tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def get_sweep_preset(preset_id: str) -> SweepPreset:
    candidate = str(preset_id).strip()
    if candidate not in _PRESETS:
        raise ValueError(f"unknown sweep preset {preset_id!r}; expected one of {', '.join(list_sweep_presets())}")
    return _PRESETS[candidate]


def build_sweep_launch_plan(
    *,
    preset_id: str,
    repo_root: Path,
    group_label: str,
    seeds: list[int],
    devices: tuple[str, ...],
    train_args: list[str] | None = None,
) -> tuple[LaunchPlan, dict[str, Any]]:
    preset = get_sweep_preset(preset_id)
    jobs: list[LaunchJob] = []
    sweep_payload_jobs: list[dict[str, Any]] = []
    stack_config = str((Path(repo_root) / "weiss_schwarz_rl" / preset.stack_config).resolve())
    job_index = 0
    for candidate in preset.candidates:
        for seed in seeds:
            override_args = list(train_args or ())
            for key, value in candidate.overrides.items():
                override_args.extend(["--config-override", f"{key}={json.dumps(value, sort_keys=True)}"])
            device = devices[job_index % len(devices)]
            jobs.append(
                LaunchJob(
                    job_id=f"{candidate.candidate_id}_seed{int(seed)}",
                    stack_config=stack_config,
                    seed=int(seed),
                    device=device,
                    run_label=f"{group_label}_{candidate.candidate_id}_seed{int(seed)}",
                    extra_args=tuple(override_args),
                )
            )
            sweep_payload_jobs.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "description": candidate.description,
                    "seed": int(seed),
                    "device": device,
                    "overrides": dict(candidate.overrides),
                }
            )
            job_index += 1
    return (
        LaunchPlan(group_label=group_label, jobs=tuple(jobs), max_parallel_jobs=len(devices)),
        {
            "kind": "compact_sweep_plan_v1",
            "preset_id": preset.preset_id,
            "group_label": group_label,
            "stack_config": preset.stack_config,
            "seeds": [int(seed) for seed in seeds],
            "devices": list(devices),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "description": candidate.description,
                    "overrides": dict(candidate.overrides),
                }
                for candidate in preset.candidates
            ],
            "jobs": sweep_payload_jobs,
        },
    )
