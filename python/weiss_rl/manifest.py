"""Run manifest schemas and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class RunManifest:
    run_id: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""
    spec_hash: str = ""
    
    #M1-11 Timescale amibiguitry elimination
    step_definition: str = "decision_boundary"  # "decision_boundary" | "learner_turn_env"
    env_wrapper: str = "DecisionBoundaryEnv"
    reward_mode: str = "shaping"  # "terminal_only" | "shaping"
    reward_perspective: str = "P"  # "P" actor-to-act | "L" fixed learning seat
    discount_gamma: float = 1.0
    
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
