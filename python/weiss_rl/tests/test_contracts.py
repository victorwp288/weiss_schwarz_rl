from __future__ import annotations

import pytest

from weiss_rl.spec import assert_spec_compatibility


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
