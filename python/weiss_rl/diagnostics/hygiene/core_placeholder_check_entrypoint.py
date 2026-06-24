from __future__ import annotations

import re
import sys
from pathlib import Path

CORE_MODULES = (
    Path("python/weiss_rl/model.py"),
    Path("python/weiss_rl/actors/actor_worker.py"),
    Path("python/weiss_rl/envs/learner_turn_env.py"),
    Path("python/weiss_rl/learners/impala/__init__.py"),
    Path("python/weiss_rl/core/spec.py"),
    Path("python/weiss_rl/config/__init__.py"),
    Path("python/weiss_rl/config/models.py"),
    Path("python/weiss_rl/config/parse.py"),
    Path("python/weiss_rl/config/hashing.py"),
    Path("python/weiss_rl/artifacts/reproducibility.py"),
    Path("python/weiss_rl/artifacts/manifest.py"),
)
PATTERN = re.compile(r"\b(?:TODO|NotImplemented(?:Error)?)\b")


def main() -> int:
    failures: list[str] = []
    for relative_path in CORE_MODULES:
        text = relative_path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                failures.append(f"{relative_path}:{line_number}: {line.strip()}")
    if failures:
        print("Core placeholder gate failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Checked {len(CORE_MODULES)} core modules: no TODO/NotImplemented placeholders found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
