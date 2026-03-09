"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.repro import canonical_json_bytes, sha256_hex

_REQUIRED_KEYS = (
    "encoding_versions",
    "action_space_size",
    "pass_id",
    "observation_dtype",
    "observation_length",
)
_HASH_KEYS = ("compatibility_hash", "spec_hash")


@dataclass(frozen=True, slots=True)
class SpecBundle:
    encoding_versions: dict[str, Any]
    action_space_size: int
    pass_id: int
    observation_dtype: str
    observation_length: int
    compatibility_hash: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _read_compatibility_hash(bundle: Mapping[str, Any]) -> str:
    for key in _HASH_KEYS:
        if key not in bundle:
            continue
        value = bundle[key]
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"{key} must be an integer or string")
        text = str(value).strip()
        if not text:
            raise ValueError(f"{key} must be non-empty")
        return text
    expected = " or ".join(_HASH_KEYS)
    raise ValueError(f"Spec bundle missing required key: {expected}")


def parse_spec_bundle(value: Mapping[str, Any]) -> SpecBundle:
    bundle = _require_mapping(value, context="Spec bundle")
    missing = [key for key in _REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"Spec bundle missing required keys: {', '.join(sorted(missing))}")

    encoding_versions = dict(_require_mapping(bundle["encoding_versions"], context="encoding_versions"))
    action_space_size = _require_int(bundle["action_space_size"], field_name="action_space_size", minimum=1)
    pass_id = _require_int(bundle["pass_id"], field_name="pass_id", minimum=0)
    observation_dtype = _require_text(bundle["observation_dtype"], field_name="observation_dtype")
    observation_length = _require_int(bundle["observation_length"], field_name="observation_length", minimum=1)
    if pass_id >= action_space_size:
        raise ValueError("pass_id must be smaller than action_space_size")

    return SpecBundle(
        encoding_versions=encoding_versions,
        action_space_size=action_space_size,
        pass_id=pass_id,
        observation_dtype=observation_dtype,
        observation_length=observation_length,
        compatibility_hash=_read_compatibility_hash(bundle),
        raw=dict(bundle),
    )


def canonical_spec_bundle_bytes(bundle: Mapping[str, Any] | SpecBundle) -> bytes:
    parsed = bundle if isinstance(bundle, SpecBundle) else parse_spec_bundle(bundle)
    return canonical_json_bytes(parsed.to_dict())


def canonical_spec_bundle_json(bundle: Mapping[str, Any] | SpecBundle) -> str:
    return canonical_spec_bundle_bytes(bundle).decode("utf-8")


def compute_spec_hash256(bundle: Mapping[str, Any] | SpecBundle) -> str:
    return sha256_hex(canonical_spec_bundle_bytes(bundle))


def assert_spec_compatibility(
    expected_spec_hash: int | str,
    observed_bundle: Mapping[str, Any] | SpecBundle,
) -> None:
    """Raise when the simulator spec hash does not match the expected value."""
    bundle = observed_bundle if isinstance(observed_bundle, SpecBundle) else parse_spec_bundle(observed_bundle)
    observed = bundle.compatibility_hash
    if str(observed) != str(expected_spec_hash):
        raise RuntimeError(
            f"Spec mismatch: expected {expected_spec_hash}, observed {observed}. "
            "Refuse to continue with mixed contracts."
        )
