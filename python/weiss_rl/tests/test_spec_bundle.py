from __future__ import annotations

import pytest

from weiss_rl.spec import (
    SpecBundle,
    canonical_spec_bundle_json,
    compute_spec_hash256,
    parse_spec_bundle,
)


def test_parse_spec_bundle_accepts_required_keys() -> None:
    bundle = parse_spec_bundle(
        {
            "encoding_versions": {"obs": 1, "action": 2},
            "action_space_size": 17,
            "pass_id": 16,
            "observation_dtype": "float32",
            "observation_length": 2048,
            "spec_hash": 8590000130,
            "extra_field": "kept-for-manifest",
        }
    )

    assert isinstance(bundle, SpecBundle)
    assert bundle.compatibility_hash == "8590000130"
    assert bundle.raw["extra_field"] == "kept-for-manifest"


@pytest.mark.parametrize("missing_key", ["pass_id", "observation_length"])
def test_parse_spec_bundle_rejects_missing_required_keys(missing_key: str) -> None:
    raw_bundle = {
        "encoding_versions": {"obs": 1},
        "action_space_size": 8,
        "pass_id": 7,
        "observation_dtype": "uint8",
        "observation_length": 64,
        "compatibility_hash": 123,
    }
    raw_bundle.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        parse_spec_bundle(raw_bundle)


def test_parse_spec_bundle_rejects_invalid_pass_id() -> None:
    with pytest.raises(ValueError, match="pass_id"):
        parse_spec_bundle(
            {
                "encoding_versions": {"obs": 1},
                "action_space_size": 3,
                "pass_id": 3,
                "observation_dtype": "uint8",
                "observation_length": 64,
                "compatibility_hash": 123,
            }
        )


def test_spec_bundle_hash_is_key_order_independent() -> None:
    left = {
        "encoding_versions": {"obs": 1, "action": 2},
        "action_space_size": 17,
        "pass_id": 16,
        "observation_dtype": "float32",
        "observation_length": 2048,
        "compatibility_hash": "8590000130",
        "extra": {"b": 2, "a": 1},
    }
    right = {
        "extra": {"a": 1, "b": 2},
        "observation_length": 2048,
        "observation_dtype": "float32",
        "pass_id": 16,
        "compatibility_hash": "8590000130",
        "action_space_size": 17,
        "encoding_versions": {"action": 2, "obs": 1},
    }

    assert canonical_spec_bundle_json(left) == (
        '{"action_space_size":17,"compatibility_hash":"8590000130","encoding_versions":'
        '{"action":2,"obs":1},"extra":{"a":1,"b":2},"observation_dtype":"float32",'
        '"observation_length":2048,"pass_id":16}'
    )
    assert compute_spec_hash256(left) == compute_spec_hash256(right)
