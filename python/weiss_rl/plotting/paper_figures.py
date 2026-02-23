"""Paper figure generation scaffold."""

from __future__ import annotations

from pathlib import Path


def render_placeholder_figure(out_path: Path) -> None:
    """Write a simple placeholder artifact until plotting is implemented."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("placeholder_figure\n", encoding="utf-8")
