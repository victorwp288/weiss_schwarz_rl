from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.manifest import RunManifest
from weiss_rl.repro import compute_seed_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    run_id = args.run_id.strip()
    print_startup_banner(args.spec_hash, args.config_hash, run_id)

    stack = load_stack_config(args.stack_config)
    print(f"Evaluation scaffold ready; seed sets: {sorted(stack.seed_sets)}")

    if run_id:
        manifest = RunManifest(
            run_id=run_id,
            config_hash=args.config_hash,
            spec_hash=args.spec_hash,
            seed_hashes=compute_seed_hashes(stack.seed_sets),
        )
        run_dir = Path("runs") / run_id
        manifest.write_json(run_dir / "manifest.json")
        print(f"Manifest written to {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
