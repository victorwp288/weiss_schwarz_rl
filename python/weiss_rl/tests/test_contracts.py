from __future__ import annotations

import pytest

from weiss_rl.spec import (
    HARD_FAIL_SPEC_MISMATCH_POLICY,
    assert_spec_compatibility,
    normalize_spec_mismatch_policy,
    require_fail_on_spec_mismatch,
)


def test_spec_compatibility_accepts_match() -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle={"spec_hash": 123})


def test_spec_compatibility_hard_fail_on_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Spec mismatch"):
        assert_spec_compatibility(expected_spec_hash=123, observed_bundle={"spec_hash": 456})


def test_normalize_spec_mismatch_policy_rejects_non_fail_fast_modes() -> None:
    with pytest.raises(ValueError, match="must be 'hard_fail'"):
        normalize_spec_mismatch_policy("warn", source="test.policy")


def test_require_fail_on_spec_mismatch_rejects_false() -> None:
    with pytest.raises(ValueError, match="must stay true"):
        require_fail_on_spec_mismatch(False, source="test.flag")


def test_fail_fast_helpers_default_to_hard_fail() -> None:
    assert normalize_spec_mismatch_policy(None, source="test.policy") == HARD_FAIL_SPEC_MISMATCH_POLICY
    assert require_fail_on_spec_mismatch(None, source="test.flag") == HARD_FAIL_SPEC_MISMATCH_POLICY
