from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_targeted_confirm_eval_resolves_explicit_or_named_seed_file(tmp_path: Path) -> None:
    from weiss_rl.eval.targeted_confirm.jobs import resolve_paired_seed_file

    explicit = tmp_path / "explicit.txt"
    explicit.write_text("1\n2\n", encoding="utf-8")
    named = tmp_path / "named.txt"
    named.write_text("3\n4\n", encoding="utf-8")
    stack = SimpleNamespace(seed_sets={"report_eval": named, "hardneg_repair_train": named})

    explicit_path, explicit_source = resolve_paired_seed_file(
        SimpleNamespace(paired_seed_file=explicit, seed_set="report_eval"),
        stack,
    )
    named_path, named_source = resolve_paired_seed_file(
        SimpleNamespace(paired_seed_file=None, seed_set="hardneg_repair_train"),
        stack,
    )

    assert explicit_path == explicit.resolve()
    assert explicit_source == "explicit"
    assert named_path == named
    assert named_source == "hardneg_repair_train"


def test_targeted_confirm_eval_resolves_main_league_sentinel_opponent_set() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import MAIN_LEAGUE_SENTINEL_OPPONENTS, resolve_opponents

    sentinel = resolve_opponents(SimpleNamespace(opponent=[], opponent_set="main_league_sentinel"))
    explicit = resolve_opponents(
        SimpleNamespace(opponent=[" B2 HeuristicPublic ", "seed_custom"], opponent_set="main_league_sentinel")
    )

    assert sentinel == MAIN_LEAGUE_SENTINEL_OPPONENTS
    assert sentinel[:2] == ["B2 HeuristicPublic", "B4 HeuristicPublicControl"]
    assert "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004" in sentinel
    assert explicit == ["B2 HeuristicPublic", "seed_custom"]


def test_targeted_confirm_eval_resolves_current_main_league_full13_opponent_set() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import MAIN_LEAGUE_FULL13_OPPONENTS, resolve_opponents

    full13 = resolve_opponents(SimpleNamespace(opponent=[], opponent_set="main_league_full13"))

    assert full13 == MAIN_LEAGUE_FULL13_OPPONENTS
    assert full13[:5] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    ]
    assert len(full13) == 13
    assert "seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025" in full13
    assert "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005" in full13


def test_targeted_confirm_eval_fast_loop_rejects_wrong_sentinel_panel() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import validate_fast_loop_eval_request

    try:
        validate_fast_loop_eval_request(
            stage="sentinel",
            paired_seeds=16,
            opponents=["B2 HeuristicPublic"],
        )
    except SystemExit as exc:
        assert "main_league_sentinel" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")


def test_targeted_confirm_eval_fast_loop_rejects_wrong_confirm_seed_depth() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import MAIN_LEAGUE_FULL13_OPPONENTS, validate_fast_loop_eval_request

    try:
        validate_fast_loop_eval_request(
            stage="confirm128",
            paired_seeds=64,
            opponents=MAIN_LEAGUE_FULL13_OPPONENTS,
        )
    except SystemExit as exc:
        assert "exactly 128 paired seeds" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")


def test_targeted_confirm_eval_fast_loop_accepts_full13_confirm64_request() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import MAIN_LEAGUE_FULL13_OPPONENTS, validate_fast_loop_eval_request

    assert (
        validate_fast_loop_eval_request(
            stage="full_confirm64",
            paired_seeds=64,
            opponents=MAIN_LEAGUE_FULL13_OPPONENTS,
        )
        is None
    )


def test_targeted_confirm_eval_fast_loop_publish_requires_full13_panel() -> None:
    from weiss_rl.eval.targeted_confirm.opponents import (
        MAIN_LEAGUE_SENTINEL_OPPONENTS,
        validate_fast_loop_eval_request,
    )

    try:
        validate_fast_loop_eval_request(
            stage="publish",
            paired_seeds=256,
            opponents=MAIN_LEAGUE_SENTINEL_OPPONENTS,
        )
    except SystemExit as exc:
        assert "main_league_full13" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")
