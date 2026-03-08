from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.manifest import RunManifest
from weiss_rl.repro import compute_seed_hashes


def _default_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    run_id = args.run_id.strip() or _default_run_id()
    print_startup_banner(args.spec_hash, args.config_hash, run_id)

    stack = load_stack_config(args.stack_config)
    print(f"Loaded stack config with {len(stack.components)} components")

    manifest = RunManifest(
        run_id=run_id,
        config_hash=args.config_hash,
        spec_hash=args.spec_hash,
        seed_hashes=compute_seed_hashes(stack.seed_sets),
        notes={
            "stack_config": str(args.stack_config),
            "component_count": len(stack.components),
            "seed_set_count": len(stack.seed_sets),
            "note": "Smoke run: config loading only (no training executed).",
        },
    )
    run_dir = Path("runs") / run_id
    manifest.write_json(run_dir / "manifest.json")
    print(f"Wrote manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
