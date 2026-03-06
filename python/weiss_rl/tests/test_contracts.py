from __future__ import annotations

from weiss_rl.spec import SpecMismatchPolicy, assert_spec_compatibility


def test_spec_compatibility_accepts_match() -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle={"spec_hash": 123})


def test_spec_compatibility_hard_fail_on_mismatch() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="Spec mismatch"):
        assert_spec_compatibility(
            expected_spec_hash=123,
            observed_bundle={"spec_hash": 456},
            policy=SpecMismatchPolicy.HARD_FAIL,
        )


def test_spec_compatibility_warn_on_mismatch() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_spec_compatibility(
            expected_spec_hash=123,
            observed_bundle={"spec_hash": 456},
            policy=SpecMismatchPolicy.WARN,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "Spec mismatch" in str(w[0].message)


def test_spec_compatibility_ignore_on_mismatch() -> None:
    # Should not raise or warn
    assert_spec_compatibility(
        expected_spec_hash=123,
        observed_bundle={"spec_hash": 456},
        policy=SpecMismatchPolicy.IGNORE,
    )
