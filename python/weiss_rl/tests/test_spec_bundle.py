from __future__ import annotations

from typing import cast

import pytest

from weiss_rl.observation_layout import parse_observation_layout_from_spec_bundle
from weiss_rl.spec import (
    SpecBundle,
    canonical_spec_bundle_json,
    compute_spec_hash256,
    parse_spec_bundle,
)


def _nested_spec_bundle(*, spec_hash: int | str = 8590000130) -> dict[str, object]:
    return {
        "policy_version": 3,
        "spec_hash": spec_hash,
        "observation": {
            "obs_encoding_version": 2,
            "dtype": "i32",
            "obs_len": 2048,
            "header_fields": [{"name": "phase", "index": 0}],
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 17,
            "pass_action_id": 16,
            "families": [{"name": "pass", "base": 16, "count": 1}],
        },
    }


def test_parse_spec_bundle_accepts_required_nested_keys() -> None:
    raw_bundle = _nested_spec_bundle()
    raw_bundle["extra_field"] = "kept-for-manifest"

    bundle = parse_spec_bundle(raw_bundle)

    assert isinstance(bundle, SpecBundle)
    assert bundle.encoding_versions == {"obs": 2, "action": 1}
    assert bundle.action_space_size == 17
    assert bundle.pass_id == 16
    assert bundle.observation_dtype == "i32"
    assert bundle.observation_length == 2048
    assert bundle.compatibility_hash == "8590000130"
    assert bundle.raw["extra_field"] == "kept-for-manifest"


@pytest.mark.parametrize(
    ("section", "field_name", "expected_message"),
    [
        ("action", "pass_action_id", "action.pass_action_id"),
        ("observation", "obs_len", "observation.obs_len"),
    ],
)
def test_parse_spec_bundle_rejects_missing_required_nested_keys(
    section: str, field_name: str, expected_message: str
) -> None:
    raw_bundle = _nested_spec_bundle(spec_hash=123)
    section_payload = dict(cast(dict[str, object], raw_bundle[section]))
    section_payload.pop(field_name)
    raw_bundle[section] = section_payload

    with pytest.raises(ValueError, match=expected_message):
        parse_spec_bundle(raw_bundle)


def test_parse_spec_bundle_rejects_invalid_pass_id() -> None:
    raw_bundle = _nested_spec_bundle(spec_hash=123)
    raw_bundle["action"] = {
        **cast(dict[str, object], raw_bundle["action"]),
        "action_space_size": 3,
        "pass_action_id": 3,
    }

    with pytest.raises(ValueError, match="action.pass_action_id"):
        parse_spec_bundle(raw_bundle)


def test_spec_bundle_hash_is_key_order_independent() -> None:
    left = {
        "policy_version": 3,
        "spec_hash": "8590000130",
        "observation": {
            "obs_encoding_version": 2,
            "dtype": "i32",
            "obs_len": 2048,
            "extra": {"b": 2, "a": 1},
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 17,
            "pass_action_id": 16,
            "families": [{"name": "pass", "count": 1, "base": 16}],
        },
    }
    right = {
        "action": {
            "families": [{"base": 16, "count": 1, "name": "pass"}],
            "pass_action_id": 16,
            "action_space_size": 17,
            "action_encoding_version": 1,
        },
        "observation": {
            "extra": {"a": 1, "b": 2},
            "obs_len": 2048,
            "dtype": "i32",
            "obs_encoding_version": 2,
        },
        "spec_hash": "8590000130",
        "policy_version": 3,
    }

    assert canonical_spec_bundle_json(left) == (
        '{"action":{"action_encoding_version":1,"action_space_size":17,"families":[{"base":16,'
        '"count":1,"name":"pass"}],"pass_action_id":16},"observation":{"dtype":"i32",'
        '"extra":{"a":1,"b":2},"obs_encoding_version":2,"obs_len":2048},'
        '"policy_version":3,"spec_hash":"8590000130"}'
    )
    assert compute_spec_hash256(left) == compute_spec_hash256(right)


def test_parse_observation_layout_from_spec_bundle_reads_typed_sections() -> None:
    layout = parse_observation_layout_from_spec_bundle(
        {
            "spec_hash": "8590000130",
            "observation": {
                "obs_encoding_version": 2,
                "dtype": "f32",
                "obs_len": 12,
                "self_first": True,
                "header_fields": [{"name": "phase", "index": 0}],
                "player_blocks": [
                    {
                        "name": "self",
                        "base": 1,
                        "len": 5,
                        "slices": [
                            {"name": "level", "start": 0, "len": 1},
                            {"name": "stage", "start": 1, "len": 4},
                        ],
                    }
                ],
                "tail_slices": [{"name": "choice_page", "start": 6, "len": 2}],
            },
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 17,
                "pass_action_id": 16,
            },
        }
    )

    assert layout.obs_len == 12
    assert layout.self_first is True
    assert layout.header_fields[0].name == "phase"
    assert layout.player_blocks[0].slices[1].indices == (2, 3, 4, 5)
    assert layout.tail_slices[0].indices == (6, 7)
