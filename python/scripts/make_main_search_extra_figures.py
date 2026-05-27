"""Export the supplementary main-search thesis figures from prepared data JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.god_search_figures import write_main_search_extra_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--figure-prefix", default="main_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_main_search_extra_figures(
        data_dir=args.data_dir,
        paper_dir=args.paper_dir,
        figure_prefix=args.figure_prefix,
    )
    print(json.dumps({key: str(value) for key, value in paths.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
