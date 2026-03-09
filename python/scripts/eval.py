from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.simulator_contract import load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    stack = load_stack_config(args.stack_config)
    reproducibility = stack.config.reproducibility
    policy = "hard_fail"
    contract = None
    if reproducibility is not None:
        spec_bundle_policy = reproducibility.spec_bundle
        policy = "hard_fail" if spec_bundle_policy.fail_on_spec_mismatch else "disabled"
        should_verify = (
            bool(args.spec_hash.strip())
            or spec_bundle_policy.require_export_spec_bundle
            or spec_bundle_policy.persist_in_manifest
        )
        if should_verify:
            contract = load_simulator_contract(stack.root)
            assert_spec_bundle_contract(args.spec_hash, contract.spec_bundle)

    print_startup_banner(args.spec_hash, args.config_hash, args.run_id, spec_mismatch_policy=policy)
    if contract is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={contract.simulator.get('compatibility_hash', '')} sha256={contract.spec_hash256}"
        )

    print(f"Evaluation scaffold ready; seed sets: {sorted(stack.seed_sets)}")


if __name__ == "__main__":
    main()
