from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config


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
    stack = load_stack_config(args.stack_config)

    print_startup_banner(
        args.spec_hash,
        args.config_hash,
        run_id,
        spec_mismatch_policy=stack.spec_mismatch_policy,
    )

    print(f"Loaded stack config with {len(stack.components)} components")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stack_config": str(args.stack_config),
        "spec_hash": args.spec_hash,
        "config_hash": args.config_hash,
        "spec_mismatch_policy": stack.spec_mismatch_policy,
        "component_count": len(stack.components),
        "seed_set_count": len(stack.seed_sets),
        "note": "Smoke run: config loading only (no training executed).",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()
