from __future__ import annotations

import pytest
from weiss_rl.core.spec import (
    HARD_FAIL_SPEC_MISMATCH_POLICY,
    assert_spec_bundle_contract,
    assert_spec_compatibility,
    canonical_json_bytes,
    normalize_bool_flag,
    normalize_spec_mismatch_policy,
    require_fail_on_spec_mismatch,
    sha256_hex,
    spec_bundle_hash,
)

_VALID_BUNDLE = {
    "encoding_versions": {"obs": 1},
    "action_space_size": 9,
    "pass_id": 8,
    "observation_dtype": "float32",
    "observation_length": 512,
    "spec_hash": 123,
}


def test_spec_compatibility_accepts_match() -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle=_VALID_BUNDLE)


def test_spec_compatibility_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="expected 124"):
        assert_spec_compatibility(expected_spec_hash=124, observed_bundle=_VALID_BUNDLE)


def test_spec_bundle_contract_accepts_bundle_sha256() -> None:
    expected_hash = sha256_hex(canonical_json_bytes(_VALID_BUNDLE))

    assert_spec_bundle_contract(expected_hash, _VALID_BUNDLE)
    assert spec_bundle_hash(_VALID_BUNDLE) == expected_hash


def test_spec_bundle_contract_rejects_bundle_sha256_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Spec bundle hash mismatch"):
        assert_spec_bundle_contract("0" * 64, _VALID_BUNDLE)


def test_normalize_spec_mismatch_policy_rejects_non_fail_fast_modes() -> None:
    with pytest.raises(ValueError, match="must be 'hard_fail'"):
        normalize_spec_mismatch_policy("warn", source="test.policy")


@pytest.mark.parametrize("value", [False, 0, 1, [], {}])
def test_normalize_spec_mismatch_policy_rejects_non_string_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be a string policy"):
        normalize_spec_mismatch_policy(value, source="test.policy")


def test_require_fail_on_spec_mismatch_rejects_false() -> None:
    with pytest.raises(ValueError, match="must stay true"):
        require_fail_on_spec_mismatch(False, source="test.flag")


def test_normalize_bool_flag_rejects_string_boolean_flags() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        normalize_bool_flag("false", source="test.flag", default=True)


def test_fail_fast_helpers_default_to_hard_fail() -> None:
    assert normalize_spec_mismatch_policy(None, source="test.policy") == HARD_FAIL_SPEC_MISMATCH_POLICY
    assert require_fail_on_spec_mismatch(None, source="test.flag") == HARD_FAIL_SPEC_MISMATCH_POLICY
