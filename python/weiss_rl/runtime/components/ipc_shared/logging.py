"""Runtime logging helpers for performance and process collector diagnostics."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_PROCESS_DEBUG_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


class PerformanceLogger:
    """Write runtime performance records as sorted JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def process_debug_log(*, run_dir: Path | None, actor_id: int, message: str) -> None:
    """Append a process-collector debug line when the environment flag is enabled."""

    if str(os.environ.get("WEISS_RL_PROCESS_DEBUG", "")).strip().lower() not in _PROCESS_DEBUG_ENABLED_VALUES:
        return
    if run_dir is None:
        return
    log_path = Path(run_dir) / "training" / "logs" / f"collector_debug_actor{int(actor_id):02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.6f} {message}\n")
