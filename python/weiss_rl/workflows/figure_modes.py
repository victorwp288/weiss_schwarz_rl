from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

RenderPublicDemoFigures = Callable[..., Mapping[str, Path]]
RenderPlaceholderFigure = Callable[[Path], None]
RenderPaperFigures = Callable[..., tuple[Path, ...]]


def run_public_demo_figure_mode(
    *,
    final_eval_dir: Path,
    out_dir: Path | None,
    render_public_demo_figures_fn: RenderPublicDemoFigures,
) -> str:
    resolved_out_dir = out_dir or (final_eval_dir.parent.parent / "figures")
    artifacts = render_public_demo_figures_fn(final_eval_dir=final_eval_dir, out_dir=resolved_out_dir)
    return f"Wrote public-demo placeholder figure bundle: {artifacts['manifest']}"


def run_placeholder_figure_mode(
    *,
    out: Path,
    render_placeholder_figure_fn: RenderPlaceholderFigure,
) -> str:
    render_placeholder_figure_fn(out)
    return f"Wrote placeholder figure: {out}"


def run_paper_figure_mode(
    *,
    run_dir: Path,
    formats: Sequence[str],
    fig_id: str | None,
    render_paper_figures_fn: RenderPaperFigures,
) -> str:
    resolved_formats = tuple(formats) if formats else ("pdf", "png")
    outputs = render_paper_figures_fn(run_dir, formats=resolved_formats, fig_id=fig_id)
    output_dir = run_dir / "figures" / "paper"
    if fig_id is None:
        return f"Wrote {len(outputs)} paper figure files to {output_dir}"
    return f"Wrote {len(outputs)} files for fig-id {fig_id!r} to {output_dir}"


def as_path_mapping(payload: Mapping[str, Any]) -> Mapping[str, Path]:
    return {str(key): Path(value) for key, value in payload.items()}
