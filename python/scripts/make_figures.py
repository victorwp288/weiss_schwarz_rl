from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.plotting.paper_figures import render_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the paper figures for a completed run directory"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing eval/ and training/ artifacts",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        help="Output format to write (repeatable). Defaults to pdf and png.",
    )
    args = parser.parse_args()

    formats = tuple(args.format) if args.format else ("pdf", "png")
    outputs = render_paper_figures(args.run_dir, formats=formats)
    print(f"Wrote {len(outputs)} paper figure files to {args.run_dir / 'figures' / 'paper'}")


if __name__ == "__main__":
    main()
