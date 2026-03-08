from __future__ import annotations

import pytest

from weiss_rl.spec import (
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


@pytest.fixture
def observed_bundle() -> dict[str, object]:
    return {
        "action_space_size": 527,
        "obs_len": 378,
        "spec_hash": 123,
    }


def test_spec_compatibility_accepts_match(observed_bundle: dict[str, object]) -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle=observed_bundle)


def test_spec_compatibility_hard_fail_on_mismatch(observed_bundle: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="Spec mismatch"):
        assert_spec_compatibility(expected_spec_hash=456, observed_bundle=observed_bundle)


def test_spec_bundle_contract_accepts_bundle_sha256(observed_bundle: dict[str, object]) -> None:
    expected_hash = sha256_hex(canonical_json_bytes(observed_bundle))

    assert_spec_bundle_contract(expected_hash, observed_bundle)
    assert spec_bundle_hash(observed_bundle) == expected_hash


def test_spec_bundle_contract_rejects_bundle_sha256_mismatch(observed_bundle: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="Spec bundle hash mismatch"):
        assert_spec_bundle_contract("0" * 64, observed_bundle)


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


def test_normalize_bool_flag_rejects_string_false() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        normalize_bool_flag("false", source="test.flag", default=True)


def test_fail_fast_helpers_default_to_hard_fail() -> None:
    assert normalize_spec_mismatch_policy(None, source="test.policy") == HARD_FAIL_SPEC_MISMATCH_POLICY
    assert require_fail_on_spec_mismatch(None, source="test.flag") == HARD_FAIL_SPEC_MISMATCH_POLICY
