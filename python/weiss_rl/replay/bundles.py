"""Replay bundle serialization scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class ReplayRecord:
    episode_key: str
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool


def write_jsonl(records: list[ReplayRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
