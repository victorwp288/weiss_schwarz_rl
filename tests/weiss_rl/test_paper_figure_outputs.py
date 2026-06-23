from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.plotting.paper_figures import PAPER_FIGURE_IDS, PAPER_FIGURE_STEMS, render_paper_figures

from .paper_figures_test_support import write_run_artifacts, write_seat_bias_artifact


def test_render_paper_figures_writes_all_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "synthetic"
    write_run_artifacts(run_dir)

    outputs = render_paper_figures(run_dir)

    output_names = {path.name for path in outputs}
    expected_names = {f"{stem}.{fmt}" for stem in PAPER_FIGURE_STEMS for fmt in ("pdf", "png")}
    assert output_names == expected_names
    assert all(path.is_file() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)


def test_render_paper_figures_can_target_single_figure_by_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "seat-bias-only"
    write_seat_bias_artifact(run_dir)

    outputs = render_paper_figures(run_dir, fig_id="seat_bias")

    assert {path.name for path in outputs} == {"fig_seat_bias.pdf", "fig_seat_bias.png"}
    assert all(path.is_file() for path in outputs)
    assert not (run_dir / "figures" / "paper" / "fig_matchup_heatmap.pdf").exists()


def test_render_paper_figures_rejects_unknown_fig_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown fig_id 'unknown'"):
        render_paper_figures(tmp_path / "runs" / "unknown-figure", fig_id="unknown")

    assert PAPER_FIGURE_IDS == (
        "matchup_heatmap",
        "truncation_heatmap",
        "seat_bias",
        "learning_curves",
    )
