"""Run manifest schemas and helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EnvWrapper = Literal["DecisionBoundaryEnv", "LearnerTurnEnv"]
StepDefinition = Literal["decision_boundary", "learner_turn_env"]
RewardMode = Literal["terminal_only_pm1", "shaping"]
RewardPerspective = Literal["P", "L"]

DEFAULT_ENV_WRAPPER: EnvWrapper = "DecisionBoundaryEnv"
DEFAULT_STEP_DEFINITION: StepDefinition = "decision_boundary"
DEFAULT_REWARD_MODE: RewardMode = "terminal_only_pm1"
DEFAULT_REWARD_PERSPECTIVE: RewardPerspective = "P"
DEFAULT_DISCOUNT_GAMMA: float = 1.0


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RunManifest:
    run_id: str
    created_at_utc: str = field(default_factory=_utc_now_isoformat)
    stack_config: str = ""
    config_hash: str = ""
    spec_hash: str = ""
    component_count: int = 0
    seed_set_count: int = 0
    env_wrapper: EnvWrapper = DEFAULT_ENV_WRAPPER
    step_definition: StepDefinition = DEFAULT_STEP_DEFINITION
    reward_mode: RewardMode = DEFAULT_REWARD_MODE
    reward_perspective: RewardPerspective = DEFAULT_REWARD_PERSPECTIVE
    discount_gamma: float = DEFAULT_DISCOUNT_GAMMA
    note: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_smoke_run_manifest(
    *,
    run_id: str,
    stack_config: Path | str,
    spec_hash: str,
    config_hash: str,
    component_count: int,
    seed_set_count: int,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        stack_config=str(stack_config),
        spec_hash=spec_hash,
        config_hash=config_hash,
        component_count=component_count,
        seed_set_count=seed_set_count,
        note="Smoke run: config loading only (no training executed).",
    )
