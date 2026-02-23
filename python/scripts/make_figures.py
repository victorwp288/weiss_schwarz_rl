from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.plotting.paper_figures import render_placeholder_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure generation scaffold entrypoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    render_placeholder_figure(args.out)
    print(f"Wrote placeholder figure: {args.out}")


if __name__ == "__main__":
    main()
