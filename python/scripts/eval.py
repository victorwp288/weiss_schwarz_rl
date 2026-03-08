from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    run_id = args.run_id.strip()
    stack = load_stack_config(args.stack_config)

    print_startup_banner(
        args.spec_hash,
        args.config_hash,
        run_id,
        spec_mismatch_policy=stack.spec_mismatch_policy,
    )

    print(f"Evaluation harness ready (spec_mismatch_policy={stack.spec_mismatch_policy})")
    print(f"Seed sets: {sorted(stack.seed_sets)}")


if __name__ == "__main__":
    main()
