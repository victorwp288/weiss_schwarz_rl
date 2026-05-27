"""Reproducibility stack config section parser."""

from __future__ import annotations

from typing import Any

from weiss_rl.core.spec import normalize_spec_mismatch_policy, require_fail_on_spec_mismatch

from .models import (
    IdsConfig,
    LegalFingerprintConfig,
    ReproducibilityConfig,
    SeedDerivationConfig,
    SpecBundlePolicyConfig,
)
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_int,
    require_mapping,
    require_str_list,
    require_text,
)


def parse_reproducibility_config(body: dict[str, Any]) -> ReproducibilityConfig:
    reject_unknown_keys(
        body,
        allowed={
            "spec_bundle",
            "ids",
            "seed_derivation",
            "seed_files",
            "determinism_requirements",
            "legal_fingerprint",
        },
        context="reproducibility",
    )
    spec_bundle = require_mapping(body["spec_bundle"], context="reproducibility.spec_bundle")
    ids = require_mapping(body["ids"], context="reproducibility.ids")
    seed_derivation = require_mapping(body["seed_derivation"], context="reproducibility.seed_derivation")
    seed_files = require_mapping(body["seed_files"], context="reproducibility.seed_files")
    legal_fingerprint = require_mapping(body["legal_fingerprint"], context="reproducibility.legal_fingerprint")
    reject_unknown_keys(
        spec_bundle,
        allowed={"require_export_spec_bundle", "persist_in_manifest", "fail_on_spec_mismatch"},
        context="reproducibility.spec_bundle",
    )
    reject_unknown_keys(
        ids,
        allowed={
            "run_id_hash",
            "config_hash",
            "spec_hash",
            "store_full_256_bit_ids",
            "store_short_64_bit_ids_for_filenames",
        },
        context="reproducibility.ids",
    )
    reject_unknown_keys(
        seed_derivation,
        allowed={"base_seed64", "actor_seed_formula", "episode_seed_formula"},
        context="reproducibility.seed_derivation",
    )
    reject_unknown_keys(
        legal_fingerprint,
        allowed={"version", "compute_in_rl_layer", "canonical_bytes", "replay_eval_mismatch_policy"},
        context="reproducibility.legal_fingerprint",
    )
    fail_on_spec_mismatch = require_bool(
        spec_bundle["fail_on_spec_mismatch"],
        field_name="reproducibility.spec_bundle.fail_on_spec_mismatch",
    )
    require_fail_on_spec_mismatch(
        fail_on_spec_mismatch,
        source="reproducibility.spec_bundle.fail_on_spec_mismatch",
    )
    replay_eval_mismatch_policy = normalize_spec_mismatch_policy(
        legal_fingerprint["replay_eval_mismatch_policy"],
        source="reproducibility.legal_fingerprint.replay_eval_mismatch_policy",
    )
    return ReproducibilityConfig(
        spec_bundle=SpecBundlePolicyConfig(
            require_export_spec_bundle=require_bool(
                spec_bundle["require_export_spec_bundle"],
                field_name="reproducibility.spec_bundle.require_export_spec_bundle",
            ),
            persist_in_manifest=require_bool(
                spec_bundle["persist_in_manifest"],
                field_name="reproducibility.spec_bundle.persist_in_manifest",
            ),
            fail_on_spec_mismatch=fail_on_spec_mismatch,
        ),
        ids=IdsConfig(
            run_id_hash=require_text(ids["run_id_hash"], field_name="reproducibility.ids.run_id_hash"),
            config_hash=require_text(ids["config_hash"], field_name="reproducibility.ids.config_hash"),
            spec_hash=require_text(ids["spec_hash"], field_name="reproducibility.ids.spec_hash"),
            store_full_256_bit_ids=require_bool(
                ids["store_full_256_bit_ids"],
                field_name="reproducibility.ids.store_full_256_bit_ids",
            ),
            store_short_64_bit_ids_for_filenames=require_bool(
                ids["store_short_64_bit_ids_for_filenames"],
                field_name="reproducibility.ids.store_short_64_bit_ids_for_filenames",
            ),
        ),
        seed_derivation=SeedDerivationConfig(
            base_seed64=require_int(
                seed_derivation["base_seed64"],
                field_name="reproducibility.seed_derivation.base_seed64",
                minimum=0,
            ),
            actor_seed_formula=require_text(
                seed_derivation["actor_seed_formula"],
                field_name="reproducibility.seed_derivation.actor_seed_formula",
            ),
            episode_seed_formula=require_text(
                seed_derivation["episode_seed_formula"],
                field_name="reproducibility.seed_derivation.episode_seed_formula",
            ),
        ),
        seed_files={
            key: require_text(value, field_name=f"reproducibility.seed_files.{key}")
            for key, value in seed_files.items()
        },
        determinism_requirements=require_str_list(
            body["determinism_requirements"],
            field_name="reproducibility.determinism_requirements",
        ),
        legal_fingerprint=LegalFingerprintConfig(
            version=require_text(legal_fingerprint["version"], field_name="reproducibility.legal_fingerprint.version"),
            compute_in_rl_layer=require_bool(
                legal_fingerprint["compute_in_rl_layer"],
                field_name="reproducibility.legal_fingerprint.compute_in_rl_layer",
            ),
            canonical_bytes=require_str_list(
                legal_fingerprint["canonical_bytes"],
                field_name="reproducibility.legal_fingerprint.canonical_bytes",
            ),
            replay_eval_mismatch_policy=replay_eval_mismatch_policy,
        ),
    )
