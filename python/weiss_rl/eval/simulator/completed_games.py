"""Completed-game records, summaries, and episode identity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from weiss_rl.artifacts.reproducibility import (
    canonical_json_bytes,
    key256_to_hex,
    key256_to_short64,
    resolve_episode_key256,
    stable_hash64,
)
from weiss_rl.core.termination_reason import classify_episode_end_reason
from weiss_rl.eval.simulator.records import EvalGameRecord, GameResult, MatchupSummary, OutcomeToken, ScheduledGame
from weiss_rl.eval.simulator.terminal_step import require_seat as _require_seat

_U32_MASK = (1 << 32) - 1


def record_completed_game(
    *,
    scheduled_game: ScheduledGame,
    result: GameResult,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
) -> EvalGameRecord:
    if int(result.episode_seed) != scheduled_game.episode_seed:
        raise ValueError(
            f"game result episode_seed mismatch: expected {scheduled_game.episode_seed}, got {int(result.episode_seed)}"
        )
    _validate_completed_game_result(result)

    episode_key256 = resolve_eval_episode_key256(scheduled_game=scheduled_game, result=result, run_id256=run_id256)
    return EvalGameRecord(
        pair_index=scheduled_game.pair_index,
        swap_index=scheduled_game.swap_index,
        episode_index=scheduled_game.episode_index,
        episode_seed=scheduled_game.episode_seed,
        episode_key=key256_to_hex(episode_key256),
        episode_key64=key256_to_short64(episode_key256),
        config_hash256=_normalize_hash256(config_hash256, name="config_hash256"),
        spec_hash256=_normalize_hash256(spec_hash256, name="spec_hash256"),
        focal_policy_id=scheduled_game.focal_policy_id,
        opponent_policy_id=scheduled_game.opponent_policy_id,
        seat0_policy_id=scheduled_game.seat0_policy_id,
        seat1_policy_id=scheduled_game.seat1_policy_id,
        focal_seat=scheduled_game.focal_seat,
        outcome=outcome_for_focal(result=result, focal_seat=scheduled_game.focal_seat),
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        engine_status=int(result.engine_status),
        seat0_deck=scheduled_game.seat0_deck,
        seat1_deck=scheduled_game.seat1_deck,
        decision_count=int(result.decision_count),
        tick_count=int(result.tick_count),
        no_progress_count=int(result.no_progress_count),
        termination_reason=_game_result_reason(result),
        total_actions=int(result.total_actions),
        pass_actions=int(result.pass_actions),
        main_move_actions=int(result.main_move_actions),
        pass_with_nonpass_available=int(result.pass_with_nonpass_available),
        max_consecutive_main_moves=int(result.max_consecutive_main_moves),
        run_id256=key256_to_hex(_coerce_run_id256(run_id256)),
    )


def outcome_for_focal(*, result: GameResult, focal_seat: int) -> OutcomeToken:
    _require_seat(focal_seat, name="focal_seat")
    if bool(result.truncated):
        return "T"
    winner_seat = result.winner_seat
    if winner_seat is None:
        return "D"
    winner = _require_seat(winner_seat, name="winner_seat")
    return "W" if winner == focal_seat else "L"


def summarize_pair_outcomes(outcomes: Sequence[str]) -> MatchupSummary:
    summary = MatchupSummary()
    for token in outcomes:
        normalized = _normalize_outcome_token(token)
        summary.games += 1
        if normalized == "W":
            summary.wins += 1
        elif normalized == "L":
            summary.losses += 1
        elif normalized == "D":
            summary.draws += 1
        else:
            summary.truncations += 1
    return summary


def summarize_game_records(records: Sequence[EvalGameRecord]) -> MatchupSummary:
    summary = summarize_pair_outcomes([record.outcome for record in records])
    for record in records:
        summary.total_actions += int(record.total_actions)
        summary.pass_actions += int(record.pass_actions)
        summary.main_move_actions += int(record.main_move_actions)
        summary.pass_with_nonpass_available += int(record.pass_with_nonpass_available)
        summary.max_consecutive_main_moves = max(
            summary.max_consecutive_main_moves,
            int(record.max_consecutive_main_moves),
        )
        if record.engine_status != 0:
            summary.engine_errors += 1
        if record.termination_reason == "no_progress_timeout":
            summary.no_progress_timeouts += 1
        elif record.termination_reason == "decision_limit_timeout":
            summary.natural_timeouts += 1
            summary.decision_limit_timeouts += 1
        elif record.termination_reason == "tick_limit_timeout":
            summary.natural_timeouts += 1
            summary.tick_limit_timeouts += 1
        elif record.termination_reason == "timeout_unknown":
            summary.natural_timeouts += 1
            summary.timeout_unknown += 1
    return summary


def write_episodes_jsonl(path: Path, records: Sequence[EvalGameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record.to_dict()).decode("utf-8"))
            handle.write("\n")


def resolve_eval_episode_key(
    *,
    scheduled_game: ScheduledGame,
    result: GameResult,
    run_id256: str | bytes,
) -> str:
    return key256_to_hex(resolve_eval_episode_key256(scheduled_game=scheduled_game, result=result, run_id256=run_id256))


def resolve_eval_episode_key256(
    *,
    scheduled_game: ScheduledGame,
    result: GameResult,
    run_id256: str | bytes,
) -> bytes:
    matchup_parts = [scheduled_game.focal_policy_id, scheduled_game.opponent_policy_id]
    if scheduled_game.seat0_deck is not None or scheduled_game.seat1_deck is not None:
        matchup_parts.extend([scheduled_game.seat0_deck or "", scheduled_game.seat1_deck or ""])
    matchup_id = "\0".join(matchup_parts)
    return resolve_episode_key256(
        simulator_episode_key=result.simulator_episode_key,
        run_id256=_coerce_run_id256(run_id256),
        actor_id=_stable_u32(scheduled_game.focal_policy_id),
        env_id=_stable_u32(matchup_id),
        episode_index=scheduled_game.episode_index,
        episode_seed64=scheduled_game.episode_seed,
    )


def _validate_completed_game_result(result: GameResult) -> None:
    terminated = bool(result.terminated)
    truncated = bool(result.truncated)
    if terminated == truncated:
        raise ValueError("completed game result must set exactly one of terminated or truncated")
    if truncated and result.winner_seat is not None:
        raise ValueError("truncated game result cannot include winner_seat")


def _game_result_reason(result: GameResult) -> str:
    if result.termination_reason is not None:
        return str(result.termination_reason)
    return classify_episode_end_reason(
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        engine_status=int(result.engine_status),
        decision_count=int(result.decision_count),
        tick_count=int(result.tick_count),
        no_progress_count=int(getattr(result, "no_progress_count", 0)),
    )


def _coerce_run_id256(run_id256: str | bytes) -> bytes:
    if isinstance(run_id256, bytes):
        if len(run_id256) != 32:
            raise ValueError(f"run_id256 must be 32 bytes, got {len(run_id256)}")
        return run_id256
    normalized = run_id256.strip()
    if len(normalized) != 64:
        raise ValueError(f"run_id256 must be 64 hex chars, got {len(normalized)}")
    return bytes.fromhex(normalized)


def _normalize_hash256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be 64 hex chars, got {len(normalized)}")
    bytes.fromhex(normalized)
    return normalized


def _stable_u32(value: str) -> int:
    return stable_hash64(value.encode("utf-8")) & _U32_MASK


def _normalize_outcome_token(token: str) -> OutcomeToken:
    normalized = token.strip().upper()
    if normalized == "W":
        return "W"
    if normalized == "L":
        return "L"
    if normalized == "D":
        return "D"
    if normalized == "T":
        return "T"
    raise ValueError(f"unknown outcome token: {token!r}")


__all__ = [
    "outcome_for_focal",
    "record_completed_game",
    "resolve_eval_episode_key",
    "resolve_eval_episode_key256",
    "summarize_game_records",
    "summarize_pair_outcomes",
    "write_episodes_jsonl",
]
