"""Export god-search comparison figures and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.god_search_figures import write_god_search_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--figure-prefix", default="god_search_k4_confirm256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_god_search_figures(
        compare_json=args.compare_json,
        out_dir=args.out_dir,
        figure_prefix=args.figure_prefix,
    )
    print(json.dumps({key: str(value) for key, value in paths.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
