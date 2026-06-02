from __future__ import annotations

from typing import Any

from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed

DEFAULT_OPPONENTS = [
    "B0 RandomLegal",
    "B1 NoLeague baseline",
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
    "seed_2aaa98fc69_seed_ae46265326_seed_bafd7a640b_policy_000011",
    "seed_2aaa98fc69_seed_ae46265326_policy_000012",
    "seed_2aaa98fc69_seed_ae46265326_policy_000014",
    "seed_2aaa98fc69_seed_ae46265326_policy_000015",
    "seed_2aaa98fc69_seed_ae46265326_policy_000016",
]

MAIN_LEAGUE_SENTINEL_OPPONENTS = [
    "B2 HeuristicPublic",
    "B4 HeuristicPublicControl",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
]
MAIN_LEAGUE_FULL13_OPPONENTS = [
    "B0 RandomLegal",
    "B1 NoLeague baseline",
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000002",
    "seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_bestresponse_u25_devbest",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
]

OPPONENT_SETS = {
    "default": DEFAULT_OPPONENTS,
    "main_league_full13": MAIN_LEAGUE_FULL13_OPPONENTS,
    "main_league_sentinel": MAIN_LEAGUE_SENTINEL_OPPONENTS,
}

FAST_LOOP_EXACT_PAIRED_SEEDS = {
    "full_confirm64": 64,
    "confirm128": 128,
    "confirm256": 256,
    "publish": 256,
}


def validate_targeted_confirm_request(args: Any) -> list[str]:
    if int(args.workers) > 1 and not bool(args.allow_parallel_workers):
        raise SystemExit(
            "targeted confirmation eval is deterministic only with --workers 1; "
            "pass --allow-parallel-workers for exploratory non-selection runs"
        )
    try:
        require_fixed_python_hash_seed("targeted confirmation eval")
    except RuntimeError as err:
        raise SystemExit(str(err)) from err
    opponents = resolve_opponents(args)
    require_fast_loop_gate(args, opponents=opponents)
    return opponents


def resolve_opponents(args: Any) -> list[str]:
    explicit = [item.strip() for item in args.opponent if item.strip()]
    if explicit:
        return explicit
    opponent_set = str(getattr(args, "opponent_set", "default"))
    try:
        return list(OPPONENT_SETS[opponent_set])
    except KeyError as exc:
        raise KeyError(f"unknown opponent set: {opponent_set}") from exc


def require_fast_loop_gate(args: Any, *, opponents: list[str]) -> None:
    stage = getattr(args, "fast_loop_stage", None)
    if not stage:
        return
    validate_fast_loop_eval_request(stage=str(stage), paired_seeds=int(args.paired_seeds), opponents=opponents)
    raise SystemExit("fast-loop gate integration was removed from the lean report workflow")


def validate_fast_loop_eval_request(*, stage: str, paired_seeds: int, opponents: list[str]) -> None:
    if stage == "sentinel":
        require_exact_opponent_panel(
            stage=stage,
            actual=opponents,
            expected=MAIN_LEAGUE_SENTINEL_OPPONENTS,
            opponent_set_name="main_league_sentinel",
        )
        return

    exact_paired_seeds = FAST_LOOP_EXACT_PAIRED_SEEDS.get(stage)
    if exact_paired_seeds is not None and paired_seeds != exact_paired_seeds:
        raise SystemExit(
            f"fast-loop stage {stage} must run exactly {exact_paired_seeds} paired seeds; "
            f"got --paired-seeds {paired_seeds}"
        )
    if stage in {"full_confirm64", "confirm128", "confirm256", "publish"}:
        require_exact_opponent_panel(
            stage=stage,
            actual=opponents,
            expected=MAIN_LEAGUE_FULL13_OPPONENTS,
            opponent_set_name="main_league_full13",
        )


def require_exact_opponent_panel(
    *,
    stage: str,
    actual: list[str],
    expected: list[str],
    opponent_set_name: str,
) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple == expected_tuple:
        return
    missing = [opponent for opponent in expected_tuple if opponent not in actual_tuple]
    extra = [opponent for opponent in actual_tuple if opponent not in expected_tuple]
    raise SystemExit(
        f"fast-loop stage {stage} must use --opponent-set {opponent_set_name} or the exact same opponent panel; "
        f"missing={missing}; extra={extra}"
    )


__all__ = [
    "DEFAULT_OPPONENTS",
    "FAST_LOOP_EXACT_PAIRED_SEEDS",
    "MAIN_LEAGUE_FULL13_OPPONENTS",
    "MAIN_LEAGUE_SENTINEL_OPPONENTS",
    "OPPONENT_SETS",
    "require_exact_opponent_panel",
    "require_fast_loop_gate",
    "resolve_opponents",
    "validate_fast_loop_eval_request",
    "validate_targeted_confirm_request",
]
