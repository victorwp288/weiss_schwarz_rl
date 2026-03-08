"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from typing import Any

HARD_FAIL_SPEC_MISMATCH_POLICY = "hard_fail"
_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(slots=True)
class RuntimeSpecBundle:
    bundle: dict[str, Any]
    spec_hash: str
    bundle_hash: str


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with stable separators and key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_bool_flag(value: object | None, *, source: str, default: bool) -> bool:
    """Return a strict boolean config value."""
    if value is None:
        return default
    if type(value) is not bool:
        raise ValueError(f"{source} must be a boolean, got {value!r}")
    return value


def normalize_spec_mismatch_policy(value: object | None, *, source: str) -> str:
    """Allow only the master-plan fail-fast policy."""
    if value is None:
        return HARD_FAIL_SPEC_MISMATCH_POLICY
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string policy, got {value!r}")

    token = value.strip().lower()
    if token != HARD_FAIL_SPEC_MISMATCH_POLICY:
        raise ValueError(
            f"{source} must be '{HARD_FAIL_SPEC_MISMATCH_POLICY}' to satisfy the fail-fast contract; "
            f"got {value!r}"
        )
    return token


def require_fail_on_spec_mismatch(value: object | None, *, source: str) -> str:
    """Require fail-fast boolean config flags to stay enabled."""
    if not normalize_bool_flag(value, source=source, default=True):
        raise ValueError(f"{source} must stay true to satisfy the fail-fast contract")
    return HARD_FAIL_SPEC_MISMATCH_POLICY


def observed_spec_hash(observed_bundle: dict[str, Any]) -> str:
    """Read the simulator compatibility hash from an exported spec bundle."""
    observed = observed_bundle.get("spec_hash", observed_bundle.get("SPEC_HASH"))
    if observed in (None, ""):
        raise RuntimeError("Observed spec bundle is missing spec_hash")
    return str(observed)


def spec_bundle_hash(observed_bundle: dict[str, Any]) -> str:
    """Compute the canonical SHA-256 of the exported runtime spec bundle."""
    return sha256_hex(canonical_json_bytes(observed_bundle))


def assert_spec_compatibility(
    expected_spec_hash: int | str,
    observed_bundle: dict[str, Any],
) -> None:
    """Raise when the simulator compatibility hash does not match the expected value."""
    observed = observed_spec_hash(observed_bundle)
    if str(observed) != str(expected_spec_hash):
        raise RuntimeError(
            f"Spec mismatch: expected {expected_spec_hash}, observed {observed}. "
            "Refuse to continue with mixed contracts."
        )


def assert_spec_bundle_contract(expected_spec_hash: str, observed_bundle: dict[str, Any]) -> None:
    """Validate either a compatibility hash or a canonical spec-bundle SHA-256."""
    token = expected_spec_hash.strip()
    if not token:
        return

    normalized = token.lower()
    if _HEX64_PATTERN.fullmatch(normalized):
        observed_bundle_hash = spec_bundle_hash(observed_bundle)
        if observed_bundle_hash != normalized:
            raise RuntimeError(
                f"Spec bundle hash mismatch: expected {normalized}, observed {observed_bundle_hash}. "
                "Refuse to continue with mixed contracts."
            )
        return

    assert_spec_compatibility(expected_spec_hash=token, observed_bundle=observed_bundle)


def should_verify_runtime_spec_bundle(
    *,
    expected_spec_hash: str,
    require_export_spec_bundle: bool,
    persist_in_manifest: bool,
) -> bool:
    """Return whether startup must export and validate the runtime spec bundle."""
    return require_export_spec_bundle or persist_in_manifest or bool(expected_spec_hash.strip())


def load_runtime_spec_bundle(*, required: bool) -> RuntimeSpecBundle | None:
    """Load the simulator's exported runtime spec bundle when available."""
    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ModuleNotFoundError as err:
        if required:
            raise RuntimeError(
                "Startup requires weiss_sim.export_spec_bundle(), but the weiss_sim module is unavailable"
            ) from err
        return None

    bundle_fn = getattr(weiss_sim, "export_spec_bundle", None)
    if not callable(bundle_fn):
        bundle_fn = getattr(weiss_sim, "spec_bundle", None)
    if not callable(bundle_fn):
        if required:
            raise RuntimeError(
                "Startup requires weiss_sim.export_spec_bundle(), but no spec-bundle export function exists"
            )
        return None

    bundle = bundle_fn()
    if not isinstance(bundle, dict):
        raise RuntimeError(f"weiss_sim.export_spec_bundle() must return a mapping, got {type(bundle).__name__}")

    return RuntimeSpecBundle(
        bundle=bundle,
        spec_hash=observed_spec_hash(bundle),
        bundle_hash=spec_bundle_hash(bundle),
    )


def verify_runtime_spec_bundle(
    expected_spec_hash: str,
    *,
    require_export_spec_bundle: bool,
    persist_in_manifest: bool,
) -> RuntimeSpecBundle | None:
    """Export the runtime spec bundle and enforce fail-fast compatibility checks."""
    required = should_verify_runtime_spec_bundle(
        expected_spec_hash=expected_spec_hash,
        require_export_spec_bundle=require_export_spec_bundle,
        persist_in_manifest=persist_in_manifest,
    )
    runtime_spec = load_runtime_spec_bundle(required=required)
    if runtime_spec is None:
        return None

    assert_spec_bundle_contract(expected_spec_hash, runtime_spec.bundle)
    return runtime_spec
