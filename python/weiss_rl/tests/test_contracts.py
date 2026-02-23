from __future__ import annotations

from weiss_rl.spec import assert_spec_compatibility


def test_spec_compatibility_accepts_match() -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle={"spec_hash": 123})
