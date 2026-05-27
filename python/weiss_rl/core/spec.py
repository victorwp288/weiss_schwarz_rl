"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, sha256_hex

HARD_FAIL_SPEC_MISMATCH_POLICY = "hard_fail"
_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_KEYS = ("compatibility_hash", "spec_hash", "SPEC_HASH")


@dataclass(slots=True)
class RuntimeSpecBundle:
    bundle: dict[str, Any]
    spec_hash: str
    bundle_hash: str


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


def normalize_bool_flag(value: object | None, *, source: str, default: bool) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise ValueError(f"{source} must be a boolean, got {value!r}")
    return value


def normalize_spec_mismatch_policy(value: object | None, *, source: str) -> str:
    if value is None:
        return HARD_FAIL_SPEC_MISMATCH_POLICY
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string policy, got {value!r}")

    policy = value.strip().lower()
    if policy != HARD_FAIL_SPEC_MISMATCH_POLICY:
        raise ValueError(
            f"{source} must be '{HARD_FAIL_SPEC_MISMATCH_POLICY}' to satisfy the fail-fast contract; got {value!r}"
        )
    return policy


def require_fail_on_spec_mismatch(value: object | None, *, source: str) -> str:
    if not normalize_bool_flag(value, source=source, default=True):
        raise ValueError(f"{source} must stay true to satisfy the fail-fast contract")
    return HARD_FAIL_SPEC_MISMATCH_POLICY


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


def _require_field(bundle: Mapping[str, Any], *, context: str, field_name: str) -> Any:
    if field_name not in bundle:
        if context == "Spec bundle":
            raise ValueError(f"Spec bundle missing required key: {field_name}")
        raise ValueError(f"Spec bundle missing required key: {context}.{field_name}")
    return bundle[field_name]


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
    action = _require_mapping(_require_field(bundle, context="Spec bundle", field_name="action"), context="action")
    observation = _require_mapping(
        _require_field(bundle, context="Spec bundle", field_name="observation"),
        context="observation",
    )

    action_encoding_version = _require_int(
        _require_field(action, context="action", field_name="action_encoding_version"),
        field_name="action.action_encoding_version",
        minimum=0,
    )
    observation_encoding_version = _require_int(
        _require_field(observation, context="observation", field_name="obs_encoding_version"),
        field_name="observation.obs_encoding_version",
        minimum=0,
    )
    action_space_size = _require_int(
        _require_field(action, context="action", field_name="action_space_size"),
        field_name="action.action_space_size",
        minimum=1,
    )
    pass_id = _require_int(
        _require_field(action, context="action", field_name="pass_action_id"),
        field_name="action.pass_action_id",
        minimum=0,
    )
    if pass_id >= action_space_size:
        raise ValueError("action.pass_action_id must be smaller than action.action_space_size")

    return SpecBundle(
        encoding_versions={"obs": observation_encoding_version, "action": action_encoding_version},
        action_space_size=action_space_size,
        pass_id=pass_id,
        observation_dtype=_require_text(
            _require_field(observation, context="observation", field_name="dtype"),
            field_name="observation.dtype",
        ),
        observation_length=_require_int(
            _require_field(observation, context="observation", field_name="obs_len"),
            field_name="observation.obs_len",
            minimum=1,
        ),
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


def observed_spec_hash(observed_bundle: Mapping[str, Any] | SpecBundle) -> str:
    if isinstance(observed_bundle, SpecBundle):
        return observed_bundle.compatibility_hash
    return _read_compatibility_hash(observed_bundle)


def spec_bundle_hash(observed_bundle: Mapping[str, Any] | SpecBundle) -> str:
    payload = observed_bundle.to_dict() if isinstance(observed_bundle, SpecBundle) else dict(observed_bundle)
    return sha256_hex(canonical_json_bytes(payload))


def assert_spec_compatibility(expected_spec_hash: int | str, observed_bundle: Mapping[str, Any] | SpecBundle) -> None:
    observed = observed_spec_hash(observed_bundle)
    if str(observed) != str(expected_spec_hash):
        raise RuntimeError(
            f"Spec mismatch: expected {expected_spec_hash}, observed {observed}. "
            "Refuse to continue with mixed contracts."
        )


def assert_spec_bundle_contract(expected_spec_hash: str, observed_bundle: Mapping[str, Any] | SpecBundle) -> None:
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
    return require_export_spec_bundle or persist_in_manifest or bool(expected_spec_hash.strip())


def load_runtime_spec_bundle(*, required: bool) -> RuntimeSpecBundle | None:
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
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f"weiss_sim.export_spec_bundle() must return a mapping, got {type(bundle).__name__}")

    try:
        parsed = parse_spec_bundle(bundle)
    except ValueError as err:
        raise RuntimeError(f"weiss_sim.export_spec_bundle() returned invalid spec bundle: {err}") from err

    payload = parsed.to_dict()
    return RuntimeSpecBundle(
        bundle=payload,
        spec_hash=parsed.compatibility_hash,
        bundle_hash=sha256_hex(canonical_json_bytes(payload)),
    )


def verify_runtime_spec_bundle(
    expected_spec_hash: str,
    *,
    require_export_spec_bundle: bool,
    persist_in_manifest: bool,
) -> RuntimeSpecBundle | None:
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
