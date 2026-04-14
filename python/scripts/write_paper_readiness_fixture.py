from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.eval.paper_readiness_fixture import write_paper_readiness_run_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a minimal thesis-grade run tree for readiness checks")
    parser.add_argument("--run-dir", type=Path, required=True, help="Destination run directory")
    args = parser.parse_args()

    run_dir = write_paper_readiness_run_fixture(args.run_dir)
    print(f"Wrote paper-readiness fixture run: {run_dir}")


if __name__ == "__main__":
    main()
