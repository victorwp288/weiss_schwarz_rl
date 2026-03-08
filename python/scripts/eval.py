from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.spec import verify_runtime_spec_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    run_id = args.run_id.strip()
    stack = load_stack_config(args.stack_config)
    runtime_spec = verify_runtime_spec_bundle(
        args.spec_hash,
        require_export_spec_bundle=stack.require_export_spec_bundle,
        persist_in_manifest=stack.persist_spec_bundle_in_manifest,
    )

    print_startup_banner(
        args.spec_hash,
        args.config_hash,
        run_id,
        spec_mismatch_policy=stack.spec_mismatch_policy,
    )
    if runtime_spec is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={runtime_spec.spec_hash} sha256={runtime_spec.bundle_hash}"
        )

    print(f"Evaluation harness ready (spec_mismatch_policy={stack.spec_mismatch_policy})")
    print(f"Seed sets: {sorted(stack.seed_sets)}")


if __name__ == "__main__":
    main()
