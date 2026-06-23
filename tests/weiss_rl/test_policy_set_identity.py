from __future__ import annotations

from importlib.util import find_spec

from weiss_rl.eval.policies.set import deck_id_for_policy_id, parse_training_policy_id


def test_deck_id_for_policy_id_locks_primary_and_heuristic_decks() -> None:
    assert deck_id_for_policy_id("policy_000021") == "preset:main_deck_5hy_yotsuba_v1"
    assert deck_id_for_policy_id("B0 RandomLegal") == "preset:main_deck_5hy_yotsuba_v1"
    assert deck_id_for_policy_id("B1 NoLeague baseline") == "preset:main_deck_5hy_yotsuba_v1"
    assert deck_id_for_policy_id("B2 HeuristicPublic") == "preset:main_deck_5hy_yotsuba_v1"
    assert deck_id_for_policy_id("B3 HeuristicPublicAggro") == "preset:aggro_deck_5hy_nino_v1"
    assert deck_id_for_policy_id("B4 HeuristicPublicControl") == "preset:control_deck_jj_s66_v1"


def test_eval_selection_compat_module_is_removed() -> None:
    assert find_spec("weiss_rl.eval.selection") is None


def test_parse_training_policy_id_matches_legacy_repo_snapshot_format() -> None:
    parsed = parse_training_policy_id("train_u50000_p3")

    assert parsed.policy_id == "train_u50000_p3"
    assert parsed.update == 50000
    assert parsed.version == 3
