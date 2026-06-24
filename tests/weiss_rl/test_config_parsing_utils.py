from __future__ import annotations

import re
from pathlib import Path

import pytest
from weiss_rl.config.loading.parsing_utils import (
    deep_merge,
    load_json,
    load_preset_document,
    load_yaml,
    reject_unknown_keys,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_int_list,
    require_mapping,
    require_str_list,
    require_text,
    resolve_repo_path,
    resolve_repo_root,
)


def test_load_yaml_and_json_require_mapping_roots(tmp_path: Path) -> None:
    yaml_path = tmp_path / "payload.yaml"
    yaml_path.write_text("alpha: 1\n", encoding="utf-8")
    assert load_yaml(yaml_path) == {"alpha": 1}

    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    assert load_yaml(empty_yaml_path) == {}

    json_path = tmp_path / "payload.json"
    json_path.write_text('{"beta": 2}', encoding="utf-8")
    assert load_json(json_path) == {"beta": 2}

    bad_yaml_path = tmp_path / "bad.yaml"
    bad_yaml_path.write_text("- alpha\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"Expected mapping in {bad_yaml_path}")):
        load_yaml(bad_yaml_path)

    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"Expected mapping in {bad_json_path}")):
        load_json(bad_json_path)


def test_scalar_require_helpers_keep_strict_type_contracts() -> None:
    assert require_mapping({"alpha": 1}, context="section") == {"alpha": 1}
    with pytest.raises(ValueError, match="section must be a mapping, got list"):
        require_mapping([], context="section")

    assert require_int(3, field_name="field", minimum=2) == 3
    with pytest.raises(ValueError, match="field must be an integer, got bool"):
        require_int(True, field_name="field")
    with pytest.raises(ValueError, match="field must be >= 2, got 1"):
        require_int(1, field_name="field", minimum=2)

    assert require_float(3, field_name="rate") == pytest.approx(3.0)
    assert require_float(1.25, field_name="rate") == pytest.approx(1.25)
    with pytest.raises(ValueError, match="rate must be numeric, got bool"):
        require_float(False, field_name="rate")

    assert require_bool(True, field_name="enabled") is True
    with pytest.raises(ValueError, match="enabled must be a boolean, got str"):
        require_bool("true", field_name="enabled")

    assert require_text("  typed_v1  ", field_name="model.encoder_kind") == "typed_v1"
    with pytest.raises(ValueError, match="model.encoder_kind must be a non-empty string"):
        require_text(" ", field_name="model.encoder_kind")


def test_collection_require_helpers_report_sorted_choices_and_member_errors() -> None:
    assert require_choice("b", field_name="kind", allowed={"c", "a", "b"}) == "b"
    with pytest.raises(ValueError, match="kind must be one of: a, b, c"):
        require_choice("z", field_name="kind", allowed={"c", "a", "b"})

    assert require_str_list([" alpha ", "beta"], field_name="names") == ("alpha", "beta")
    with pytest.raises(ValueError, match=r"names\[\] must be a non-empty string"):
        require_str_list(["alpha", ""], field_name="names")

    assert require_int_list([0, 3], field_name="ids") == (0, 3)
    with pytest.raises(ValueError, match=r"ids\[\] must be an integer, got bool"):
        require_int_list([True], field_name="ids")
    with pytest.raises(ValueError, match=r"ids\[\] must be >= 0, got -1"):
        require_int_list([-1], field_name="ids")


def test_reject_unknown_keys_reports_sorted_keys() -> None:
    reject_unknown_keys({"alpha": 1}, allowed={"alpha"}, context="section")
    with pytest.raises(ValueError, match="section has unsupported keys: beta, gamma"):
        reject_unknown_keys({"gamma": 3, "alpha": 1, "beta": 2}, allowed={"alpha"}, context="section")


def test_deep_merge_recurses_without_mutating_inputs() -> None:
    base = {"model": {"width": 128, "dropout": {"family_a": 0.1}}, "extends": "ignored.yaml"}
    overlay = {"model": {"dropout": {"family_a": 0.0}, "depth": 2}, "extends": "parent.yaml"}

    merged = deep_merge(base, overlay)

    assert merged == {"model": {"width": 128, "dropout": {"family_a": 0.0}, "depth": 2}, "extends": "ignored.yaml"}
    assert base == {"model": {"width": 128, "dropout": {"family_a": 0.1}}, "extends": "ignored.yaml"}
    assert overlay == {"model": {"dropout": {"family_a": 0.0}, "depth": 2}, "extends": "parent.yaml"}


def test_load_preset_document_applies_extends_and_detects_cycles(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "schema_version: 2\nmodel:\n  encoder_kind: typed_v1\n  width: 128\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: parent.yaml\nmodel:\n  width: 256\ntraining:\n  algorithm: impala_vtrace_gru\n",
        encoding="utf-8",
    )

    assert load_preset_document(child) == {
        "schema_version": 2,
        "model": {"encoder_kind": "typed_v1", "width": 256},
        "training": {"algorithm": "impala_vtrace_gru"},
    }

    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text("extends: right.yaml\n", encoding="utf-8")
    right.write_text("extends: left.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Config extends cycle detected"):
        load_preset_document(left)


def test_load_preset_document_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    preset = tmp_path / "bad.yaml"
    preset.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"{preset.resolve()} has unsupported keys: unknown")):
        load_preset_document(preset)


def test_repo_path_resolution_helpers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    configs = repo / "configs"
    nested = configs / "presets"
    nested.mkdir(parents=True)
    stack_file = nested / "local.yaml"
    stack_file.write_text("schema_version: 2\n", encoding="utf-8")

    assert resolve_repo_root(stack_file) == repo
    assert resolve_repo_path(repo, "configs/seeds/dev.txt") == repo / "configs" / "seeds" / "dev.txt"

    absolute = tmp_path / "absolute.txt"
    assert resolve_repo_path(repo, str(absolute)) == absolute

    with pytest.raises(FileNotFoundError, match="Could not resolve repo root"):
        resolve_repo_root(tmp_path / "outside.yaml")
