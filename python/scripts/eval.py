from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.config import load_stack_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    args = parser.parse_args()

    stack = load_stack_config(args.stack_config)
    print(f"Evaluation scaffold ready; seed sets: {sorted(stack.seed_sets)}")


if __name__ == "__main__":
    main()
