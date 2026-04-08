from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.plotting.paper_figures import render_placeholder_figure, render_public_demo_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder figure writer (not the final paper figure pipeline)")
    parser.add_argument("--out", type=Path, default=None, help="Legacy single placeholder output path")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Render clearly-labeled demo-only placeholder figures from public-demo final_eval artifacts.",
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
        if args.out is not None:
            parser.error("--public-demo cannot be combined with --out")
        out_dir = args.out_dir or (args.final_eval_dir.parent.parent / "figures")
        artifacts = render_public_demo_figures(final_eval_dir=args.final_eval_dir, out_dir=out_dir)
        print(f"Wrote public-demo placeholder figure bundle: {artifacts['manifest']}")
        return

    if args.out is None:
        parser.error("--out is required unless --public-demo is used")

    render_placeholder_figure(args.out)
    print(f"Wrote placeholder figure: {args.out}")


if __name__ == "__main__":
    main()
