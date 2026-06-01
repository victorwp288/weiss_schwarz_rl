"""Runtime orchestration for trajectory policy-drift diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed
from weiss_rl.config import load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    build_trajectory_policy_drift_report,
    parse_policy_specs,
    print_trajectory_policy_drift_summary,
    write_trajectory_policy_drift_report,
)
from weiss_rl.experiments.trajectory_policy_drift_scoring import (
    action_catalog_from_stack_spec,
    configure_torch_determinism,
    family_metadata,
    score_policy,
)
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset


def run_trajectory_policy_drift(args: argparse.Namespace) -> dict[str, Any]:
    python_hash_seed = require_fixed_python_hash_seed("trajectory_policy_drift")
    dataset = load_replay_trajectory_bc_dataset(args.dataset)
    if int(dataset.metadata.get("train_rows", 0)) <= 0:
        raise SystemExit("dataset has no trainable rows")
    stack = load_stack_config(args.stack_config)
    spec_hash = str(dataset.metadata.get("spec_hash256") or "").strip()
    if not spec_hash:
        raise SystemExit("dataset metadata is missing spec_hash256")
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    action_catalog = action_catalog_from_stack_spec(contract.spec_bundle)
    family_names, family_by_action = family_metadata(action_catalog=action_catalog, action_dim=action_dim)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    configure_torch_determinism(torch_threads=int(args.torch_threads))
    policy_specs = parse_policy_specs(args.policy)
    reference_label = str(args.reference_label or policy_specs[0].label)

    scores_by_label = {
        spec.label: score_policy(
            spec=spec,
            stack=stack,
            dataset=dataset,
            contract_spec_bundle=contract.spec_bundle,
            action_dim=action_dim,
            family_by_action=family_by_action,
            device=device,
        )
        for spec in policy_specs
    }
    report = build_trajectory_policy_drift_report(
        stack_config=Path(args.stack_config),
        dataset_path=Path(args.dataset),
        dataset=dataset,
        device=device,
        python_hash_seed=int(python_hash_seed),
        torch_threads=int(args.torch_threads),
        output_json=Path(args.output_json),
        reference_label=reference_label,
        policy_specs=policy_specs,
        scores_by_label=scores_by_label,
        family_names=family_names,
        family_by_action=family_by_action,
        max_examples=int(args.max_examples),
    )
    write_trajectory_policy_drift_report(Path(args.output_json), report)
    print_trajectory_policy_drift_summary(report)
    return report
