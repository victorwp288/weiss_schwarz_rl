from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import StackConfig, canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.manifest import RunManifest, build_seed_file_manifest, default_run_dir_name, write_run_artifacts
from weiss_rl.repro import compute_run_id64, compute_run_id256
from weiss_rl.simulator_contract import load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract

_SHA256_HEX_LENGTH = 64
_U64_MASK = (1 << 64) - 1


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    return result.stdout.strip()


def _git_commit() -> str:
    try:
        return _git_output(["rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_dirty() -> bool:
    try:
        return bool(_git_output(["status", "--short"]))
    except (OSError, subprocess.CalledProcessError):
        return False


def _start_nonce() -> int:
    return time.time_ns() & _U64_MASK


def _hardware_summary() -> dict[str, str | int]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
    }


def _evaluation_pinning(stack: StackConfig) -> dict[str, str | bool]:
    if stack.config.evaluation is None:
        return {}
    evaluation = stack.config.evaluation
    return {
        "eval_device": evaluation.eval_device,
        "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
        "eval_inference_mode": evaluation.eval_inference_mode,
        "seat_swap": evaluation.seat_swap,
        "legal_fingerprint_version": evaluation.legal_fingerprint_checks.version,
        "legal_fingerprint_mismatch_policy": evaluation.legal_fingerprint_checks.mismatch_policy,
    }


def _policy_set_selection(stack: StackConfig) -> list[str]:
    if stack.config.evaluation is None:
        return []
    selection = stack.config.evaluation.final_policy_set_selection
    return [*selection.fixed_anchor_set_v1.required, *selection.fixed_anchor_set_v1.optional_if_available]


def _spec_mismatch_policy(stack: StackConfig) -> str:
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return "hard_fail"
    return "hard_fail" if reproducibility.spec_bundle.fail_on_spec_mismatch else "disabled"


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifest/provenance smoke entrypoint (not an end-to-end training loop)"
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        required=True,
        help="Path to the stack config used for the manifest/provenance scaffold",
    )
    parser.add_argument(
        "--spec-hash",
        type=str,
        default="",
        help="Expected compatibility spec hash or full spec bundle SHA-256",
    )
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    stack = load_stack_config(args.stack_config)
    simulator_contract = load_simulator_contract(stack.root)
    assert_spec_bundle_contract(args.spec_hash, simulator_contract.spec_bundle)

    spec_hash256 = simulator_contract.spec_hash256
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = _git_commit()
    start_nonce = _start_nonce()
    run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
    run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
    run_dir_name = run_label or default_run_dir_name(run_id64)

    print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label,
        run_dir_name=run_dir_name,
        spec_mismatch_policy=_spec_mismatch_policy(stack),
    )
    print(
        "Verified runtime spec bundle: "
        f"compat={simulator_contract.simulator.get('compatibility_hash', '')} sha256={spec_hash256}"
    )
    print(f"Loaded stack config with {len(stack.components)} components")

    manifest = RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=_git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_contract.simulator,
        spec_bundle=simulator_contract.spec_bundle,
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=_hardware_summary(),
        evaluation_pinning=_evaluation_pinning(stack),
        policy_set_selection=_policy_set_selection(stack),
    )
    artifacts = write_run_artifacts(
        stack.root / "runs",
        manifest,
        run_label=run_label or None,
    )
    print(f"Wrote manifest: {artifacts.manifest_path}")
    print("Manifest scaffold only: no learner training or rollout collection was executed.")


if __name__ == "__main__":
    main()
