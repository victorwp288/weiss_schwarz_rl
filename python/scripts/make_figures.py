from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.plotting.paper_figures import (
    PAPER_FIGURE_IDS,
    render_paper_figures,
    render_placeholder_figure,
    render_public_demo_figures,
)


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
        choices=PAPER_FIGURE_IDS,
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

        out_dir = args.out_dir or (args.final_eval_dir.parent.parent / "figures")
        artifacts = render_public_demo_figures(final_eval_dir=args.final_eval_dir, out_dir=out_dir)
        print(f"Wrote public-demo placeholder figure bundle: {artifacts['manifest']}")
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
        render_placeholder_figure(args.out)
        print(f"Wrote placeholder figure: {args.out}")
        return

    if args.run_dir is None:
        parser.error("--run-dir is required unless --public-demo or --out is used")
    if args.final_eval_dir is not None or args.out_dir is not None:
        parser.error("--final-eval-dir and --out-dir require --public-demo")

    formats = tuple(args.format) if args.format else ("pdf", "png")
    outputs = render_paper_figures(args.run_dir, formats=formats, fig_id=args.fig_id)
    if args.fig_id is None:
        print(f"Wrote {len(outputs)} paper figure files to {args.run_dir / 'figures' / 'paper'}")
    else:
        print(
            f"Wrote {len(outputs)} files for fig-id {args.fig_id!r} to "
            f"{args.run_dir / 'figures' / 'paper'}"
        )


if __name__ == "__main__":
    main()
