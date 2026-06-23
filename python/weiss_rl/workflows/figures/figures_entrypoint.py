from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.plotting import paper_figures as _paper_figures
from weiss_rl.workflows.figures import figure_modes as _figure_modes

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper figures or public-safe demo figures")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing eval/ and training/ artifacts",
    )
    parser.add_argument(
        "--fig-id",
        choices=_paper_figures.PAPER_FIGURE_IDS,
        help="Stable figure ID to render. Defaults to rendering all paper figures.",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        help="Output format to write (repeatable). Defaults to pdf and png.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Legacy single placeholder output path")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Render clearly-labeled demo-only figures from public-demo final_eval artifacts.",
    )
    parser.add_argument(
        "--final-eval-dir",
        type=Path,
        default=None,
        help="Input final_eval directory for --public-demo mode",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for --public-demo mode (default: <final-eval-dir>/../../figures)",
    )
    args = parser.parse_args()

    if args.public_demo:
        if args.final_eval_dir is None:
            parser.error("--public-demo requires --final-eval-dir")
        if args.run_dir is not None:
            parser.error("--public-demo cannot be combined with --run-dir")
        if args.fig_id is not None:
            parser.error("--public-demo cannot be combined with --fig-id")
        if args.format is not None:
            parser.error("--public-demo cannot be combined with --format")
        if args.out is not None:
            parser.error("--public-demo cannot be combined with --out")

        print(
            _figure_modes.run_public_demo_figure_mode(
                final_eval_dir=args.final_eval_dir,
                out_dir=args.out_dir,
                render_public_demo_figures_fn=_paper_figures.render_public_demo_figures,
            )
        )
        return

    if args.out is not None:
        if args.run_dir is not None:
            parser.error("--out cannot be combined with --run-dir")
        if args.fig_id is not None:
            parser.error("--out cannot be combined with --fig-id")
        if args.format is not None:
            parser.error("--out cannot be combined with --format")
        if args.final_eval_dir is not None or args.out_dir is not None:
            parser.error("--out cannot be combined with --final-eval-dir or --out-dir")
        print(
            _figure_modes.run_placeholder_figure_mode(
                out=args.out,
                render_placeholder_figure_fn=_paper_figures.render_placeholder_figure,
            )
        )
        return

    if args.run_dir is None:
        parser.error("--run-dir is required unless --public-demo or --out is used")
    if args.final_eval_dir is not None or args.out_dir is not None:
        parser.error("--final-eval-dir and --out-dir require --public-demo")

    print(
        _figure_modes.run_paper_figure_mode(
            run_dir=args.run_dir,
            formats=tuple(args.format or ()),
            fig_id=args.fig_id,
            render_paper_figures_fn=_paper_figures.render_paper_figures,
        )
    )


if __name__ == "__main__":
    main()
