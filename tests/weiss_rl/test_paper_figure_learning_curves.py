from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.plotting.paper_figures import render_paper_figures

from .paper_figures_test_support import write_run_artifacts


def test_render_paper_figures_uses_interpolation_provenance_when_training_metrics_are_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "interpolated"
    write_run_artifacts(run_dir)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").unlink()
    interpolation_dir = run_dir / "eval" / "diagnostics"
    interpolation_dir.mkdir(parents=True, exist_ok=True)
    (interpolation_dir / "checkpoint_interpolation_summary.json").write_text(
        json.dumps(
            {
                "first_checkpoint": "runs/source_a/training/checkpoints/checkpoint_10.pt",
                "second_checkpoint": "runs/source_b/training/checkpoints/checkpoint_5.pt",
                "second_weight": 0.15,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = render_paper_figures(run_dir, fig_id="learning_curves")

    assert {path.name for path in outputs} == {"fig_learning_curves.pdf", "fig_learning_curves.png"}
    assert all(path.is_file() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)
