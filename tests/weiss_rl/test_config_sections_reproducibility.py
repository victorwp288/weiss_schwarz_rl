from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from weiss_rl.config.sections_reproducibility import parse_reproducibility_config


def _copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def _reproducibility_body() -> dict[str, object]:
    return {
        "spec_bundle": {
            "require_export_spec_bundle": True,
            "persist_in_manifest": True,
            "fail_on_spec_mismatch": True,
        },
        "ids": {
            "run_id_hash": "run_id256",
            "config_hash": "config_hash256",
            "spec_hash": "spec_hash256",
            "store_full_256_bit_ids": True,
            "store_short_64_bit_ids_for_filenames": True,
        },
        "seed_derivation": {
            "base_seed64": 123,
            "actor_seed_formula": "base+actor",
            "episode_seed_formula": "base+episode",
        },
        "seed_files": {
            "dev_eval": "configs/seeds/dev_eval_seeds.txt",
            "final_eval": "configs/seeds/final_eval_seeds.txt",
        },
        "determinism_requirements": ["cpu_eval", "seat_swap"],
        "legal_fingerprint": {
            "version": "legal_fingerprint_v1",
            "compute_in_rl_layer": True,
            "canonical_bytes": ["ids", "families"],
            "replay_eval_mismatch_policy": "hard_fail",
        },
    }


def test_parse_reproducibility_config_accepts_fail_fast_contract() -> None:
    config = parse_reproducibility_config(_reproducibility_body())

    assert config.spec_bundle.require_export_spec_bundle is True
    assert config.spec_bundle.persist_in_manifest is True
    assert config.spec_bundle.fail_on_spec_mismatch is True
    assert config.ids.run_id_hash == "run_id256"
    assert config.seed_derivation.base_seed64 == 123
    assert config.seed_files["dev_eval"] == "configs/seeds/dev_eval_seeds.txt"
    assert config.determinism_requirements == ("cpu_eval", "seat_swap")
    assert config.legal_fingerprint.replay_eval_mismatch_policy == "hard_fail"


def test_parse_reproducibility_config_preserves_fail_fast_errors() -> None:
    bad_spec = _reproducibility_body()
    bad_spec["spec_bundle"] = {
        "require_export_spec_bundle": True,
        "persist_in_manifest": True,
        "fail_on_spec_mismatch": False,
    }
    with pytest.raises(
        ValueError,
        match="reproducibility.spec_bundle.fail_on_spec_mismatch must stay true",
    ):
        parse_reproducibility_config(bad_spec)

    bad_policy = _reproducibility_body()
    legal = _copy_section(bad_policy, "legal_fingerprint")
    legal["replay_eval_mismatch_policy"] = "warn"
    bad_policy["legal_fingerprint"] = legal
    with pytest.raises(
        ValueError,
        match="reproducibility.legal_fingerprint.replay_eval_mismatch_policy must be 'hard_fail'",
    ):
        parse_reproducibility_config(bad_policy)


def test_parse_reproducibility_config_preserves_unknown_key_errors() -> None:
    unknown = _reproducibility_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="reproducibility has unsupported keys: extra"):
        parse_reproducibility_config(unknown)

    bad_ids = _reproducibility_body()
    ids = _copy_section(bad_ids, "ids")
    ids["extra"] = True
    bad_ids["ids"] = ids
    with pytest.raises(ValueError, match="reproducibility.ids has unsupported keys: extra"):
        parse_reproducibility_config(bad_ids)

    bad_legal = _reproducibility_body()
    legal = _copy_section(bad_legal, "legal_fingerprint")
    legal["extra"] = True
    bad_legal["legal_fingerprint"] = legal
    with pytest.raises(ValueError, match="reproducibility.legal_fingerprint has unsupported keys: extra"):
        parse_reproducibility_config(bad_legal)


def test_parse_reproducibility_config_preserves_type_and_minimum_errors() -> None:
    bad_flag = _reproducibility_body()
    spec_bundle = _copy_section(bad_flag, "spec_bundle")
    spec_bundle["persist_in_manifest"] = "true"
    bad_flag["spec_bundle"] = spec_bundle
    with pytest.raises(ValueError, match="reproducibility.spec_bundle.persist_in_manifest must be a boolean, got str"):
        parse_reproducibility_config(bad_flag)

    bad_seed = _reproducibility_body()
    seed_derivation = _copy_section(bad_seed, "seed_derivation")
    seed_derivation["base_seed64"] = -1
    bad_seed["seed_derivation"] = seed_derivation
    with pytest.raises(ValueError, match="reproducibility.seed_derivation.base_seed64 must be >= 0, got -1"):
        parse_reproducibility_config(bad_seed)

    bad_requirements = _reproducibility_body()
    bad_requirements["determinism_requirements"] = ["cpu_eval", ""]
    with pytest.raises(ValueError, match=r"reproducibility.determinism_requirements\[\] must be a non-empty string"):
        parse_reproducibility_config(bad_requirements)
