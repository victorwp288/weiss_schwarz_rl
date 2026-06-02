from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.core.spec import assert_spec_bundle_contract
from weiss_rl.diagnostics.cli_banner import print_startup_banner
from weiss_rl.experiments.toy_public_demo import public_demo_spec_bundle, public_demo_spec_hash256
from weiss_rl.workflows.eval_support.eval_reports import _expected_sha256, _require_matching_hash


@dataclass(frozen=True, slots=True)
class EvalStartupDependencies:
    load_stack_config_fn: Any = load_stack_config
    compute_config_hash256_fn: Any = compute_config_hash256
    expected_sha256_fn: Any = _expected_sha256
    require_matching_hash_fn: Any = _require_matching_hash
    public_demo_spec_bundle_fn: Any = public_demo_spec_bundle
    assert_spec_bundle_contract_fn: Any = assert_spec_bundle_contract
    public_demo_spec_hash256_fn: Any = public_demo_spec_hash256
    load_verified_simulator_contract_fn: Any = load_verified_simulator_contract
    print_startup_banner_fn: Any = print_startup_banner


def build_eval_startup_dependencies(entrypoint_globals: Mapping[str, Any]) -> EvalStartupDependencies:
    return EvalStartupDependencies(
        load_stack_config_fn=entrypoint_globals["load_stack_config"],
        compute_config_hash256_fn=entrypoint_globals["compute_config_hash256"],
        expected_sha256_fn=entrypoint_globals["_expected_sha256"],
        require_matching_hash_fn=entrypoint_globals["_require_matching_hash"],
        public_demo_spec_bundle_fn=entrypoint_globals["public_demo_spec_bundle"],
        assert_spec_bundle_contract_fn=entrypoint_globals["assert_spec_bundle_contract"],
        public_demo_spec_hash256_fn=entrypoint_globals["public_demo_spec_hash256"],
        load_verified_simulator_contract_fn=entrypoint_globals["load_verified_simulator_contract"],
        print_startup_banner_fn=entrypoint_globals["print_startup_banner"],
    )


__all__ = ["EvalStartupDependencies", "build_eval_startup_dependencies"]
