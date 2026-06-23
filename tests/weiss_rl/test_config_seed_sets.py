from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.config.seed_sets import parse_seed_sets_override, resolve_seed_sets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_resolve_seed_sets_preserves_source_precedence() -> None:
    stack = load_stack_config(_repo_root() / "configs/presets/structured_acceptance_standard.yaml")
    assert stack.config.league is not None
    assert stack.config.evaluation is not None
    assert stack.config.reproducibility is not None

    league = replace(
        stack.config.league,
        promotion=replace(stack.config.league.promotion, seed_file="configs/seeds/from_league.txt"),
    )
    evaluation = replace(
        stack.config.evaluation,
        seed_files={
            "shared": "configs/seeds/from_eval.txt",
            "promotion_gate": "configs/seeds/from_eval_promotion.txt",
        },
    )
    reproducibility = replace(
        stack.config.reproducibility,
        seed_files={
            "shared": "configs/seeds/from_repro.txt",
            "promotion_gate": "configs/seeds/from_repro_promotion.txt",
            "repro_only": "configs/seeds/from_repro_only.txt",
        },
    )

    seed_sets = resolve_seed_sets(
        root=stack.root,
        league=league,
        evaluation=evaluation,
        reproducibility=reproducibility,
    )

    assert seed_sets["shared"] == stack.root / "configs/seeds/from_eval.txt"
    assert seed_sets["promotion_gate"] == stack.root / "configs/seeds/from_eval_promotion.txt"
    assert seed_sets["repro_only"] == stack.root / "configs/seeds/from_repro_only.txt"


def test_resolve_seed_sets_uses_league_promotion_when_eval_omits_it() -> None:
    stack = load_stack_config(_repo_root() / "configs/presets/structured_acceptance_standard.yaml")
    assert stack.config.league is not None
    assert stack.config.evaluation is not None
    assert stack.config.reproducibility is not None

    league = replace(
        stack.config.league,
        promotion=replace(stack.config.league.promotion, seed_file="configs/seeds/from_league.txt"),
    )
    evaluation = replace(stack.config.evaluation, seed_files={})
    reproducibility = replace(
        stack.config.reproducibility,
        seed_files={"promotion_gate": "configs/seeds/from_repro_promotion.txt"},
    )

    seed_sets = resolve_seed_sets(
        root=stack.root,
        league=league,
        evaluation=evaluation,
        reproducibility=reproducibility,
    )

    assert seed_sets["promotion_gate"] == stack.root / "configs/seeds/from_league.txt"


def test_resolve_seed_sets_skips_blank_league_seed_file() -> None:
    stack = load_stack_config(_repo_root() / "configs/presets/structured_acceptance_standard.yaml")
    assert stack.config.league is not None
    assert stack.config.reproducibility is not None

    league = replace(stack.config.league, promotion=replace(stack.config.league.promotion, seed_file=" "))
    reproducibility = replace(
        stack.config.reproducibility,
        seed_files={"promotion_gate": "configs/seeds/from_repro_promotion.txt"},
    )

    seed_sets = resolve_seed_sets(root=stack.root, league=league, evaluation=None, reproducibility=reproducibility)

    assert seed_sets["promotion_gate"] == stack.root / "configs/seeds/from_repro_promotion.txt"


def test_parse_seed_sets_override_resolves_paths_and_validates_text(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    absolute = tmp_path / "absolute.txt"

    seed_sets = parse_seed_sets_override(
        root=root,
        seed_sets_doc={"dev_eval": "configs/seeds/dev.txt", "absolute": str(absolute)},
    )

    assert seed_sets["dev_eval"] == root / "configs/seeds/dev.txt"
    assert seed_sets["absolute"] == absolute

    with pytest.raises(ValueError, match="seed_sets.<key> must be a non-empty string"):
        parse_seed_sets_override(root=root, seed_sets_doc=cast(Any, {1: "configs/seeds/dev.txt"}))

    with pytest.raises(ValueError, match="seed_sets.dev_eval must be a non-empty string"):
        parse_seed_sets_override(root=root, seed_sets_doc={"dev_eval": ""})
