from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import StackConfig, canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.manifest import RunManifest, build_seed_file_manifest, write_run_artifacts
from weiss_rl.repro import compute_run_id64, compute_run_id256
from weiss_rl.simulator_contract import load_simulator_contract

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Expected spec_hash256 for contract validation")
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run directory label override")
    args = parser.parse_args()

    stack = load_stack_config(args.stack_config)
    simulator_contract = load_simulator_contract(stack.root)
    spec_hash256 = simulator_contract.spec_hash256
    config_hash256 = compute_config_hash256(stack)

    _require_matching_hash(
        flag_name="--spec-hash",
        expected=_expected_sha256(args.spec_hash, flag_name="--spec-hash"),
        actual=spec_hash256,
    )
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = _git_commit()
    start_nonce = _start_nonce()
    run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
    run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"

    print_startup_banner(spec_hash256, config_hash256, run_id64)
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
        run_dir_name=args.run_id.strip() or None,
    )
    print(f"Wrote manifest: {artifacts.manifest_path}")


if __name__ == "__main__":
    main()
