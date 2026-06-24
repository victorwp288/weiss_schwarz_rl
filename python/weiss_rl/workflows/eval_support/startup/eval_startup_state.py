from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalValidatedArgs:
    run_label: str
    paired_seed_limit: int | None
    stage1_paired_seeds: int | None
    max_paired_seeds: int | None


@dataclass(frozen=True, slots=True)
class EvalStartup:
    stack: Any
    config_hash256: str
    reported_spec_hash: str
    contract: Any | None
