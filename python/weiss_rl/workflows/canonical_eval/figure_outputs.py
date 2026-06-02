from __future__ import annotations

from pathlib import Path
from typing import Any


def build_canonical_figure_outputs(*, run_dir: Path, dependencies: Any) -> tuple[Path, ...]:
    return dependencies.render_paper_figures_fn(run_dir)


__all__ = ["build_canonical_figure_outputs"]
