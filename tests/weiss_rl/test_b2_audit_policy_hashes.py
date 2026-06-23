from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.diagnostics import b2_disagreement_audit as audit_module


def test_resolve_requested_policy_id_accepts_registry_alias_for_train_policy_id() -> None:
    resolved = audit_module._resolve_requested_policy_id(
        requested_policy_id="policy_000015",
        source_focal_policy_id="train_u300_p15",
    )

    assert resolved == "policy_000015"


def test_inspection_policy_id_maps_b1_display_id_to_registry_alias() -> None:
    assert audit_module._inspection_policy_id("B1 NoLeague baseline") == "b1_noleague_baseline"
    assert audit_module._inspection_policy_id("B2 HeuristicPublic") == "B2 HeuristicPublic"


def test_run_config_hashes_reads_b1_hash_from_hash_file_and_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "b1"
    run_dir.mkdir()
    run_dir.joinpath("config_hash256.txt").write_text("a" * 64 + "\n", encoding="utf-8")
    run_dir.joinpath("manifest.json").write_text(json.dumps({"config_hash256": "b" * 64}), encoding="utf-8")

    assert audit_module._run_config_hashes(run_dir) == ["a" * 64, "b" * 64]


def test_resolve_requested_policy_id_requires_explicit_mismatch_mode() -> None:
    rejected = audit_module._resolve_requested_policy_id(
        requested_policy_id="selected_seed",
        source_focal_policy_id="policy_000005",
    )
    accepted = audit_module._resolve_requested_policy_id(
        requested_policy_id="selected_seed",
        source_focal_policy_id="policy_000005",
        allow_mismatch=True,
    )

    assert rejected is None
    assert accepted == "selected_seed"


def test_resolve_source_config_hash_accepts_run_manifest_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.joinpath("manifest.json").write_text(
        json.dumps({"config_hash256": "b" * 64}),
        encoding="utf-8",
    )

    resolved, manifest_hash = audit_module._resolve_source_config_hash(
        source_config_hash256="b" * 64,
        stack_config_hash256="a" * 64,
        run_dir=run_dir,
    )

    assert resolved == "b" * 64
    assert manifest_hash == "b" * 64


def test_resolve_source_config_hash_rejects_unmatched_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stack config hash does not match"):
        audit_module._resolve_source_config_hash(
            source_config_hash256="c" * 64,
            stack_config_hash256="a" * 64,
            run_dir=tmp_path / "missing_run",
        )


def test_audit_run_id_includes_opponent_policy_id(tmp_path: Path) -> None:
    b2_id = audit_module._audit_run_id256(
        policy_id="policy_000002",
        opponent_policy_id="B2 HeuristicPublic",
        episodes_jsonl=tmp_path / "episodes.jsonl",
        output_run_dir=tmp_path / "out",
        paired_seeds=(1, 2),
    )
    b3_id = audit_module._audit_run_id256(
        policy_id="policy_000002",
        opponent_policy_id="B3 HeuristicPublicAggro",
        episodes_jsonl=tmp_path / "episodes.jsonl",
        output_run_dir=tmp_path / "out",
        paired_seeds=(1, 2),
    )

    assert b2_id != b3_id
