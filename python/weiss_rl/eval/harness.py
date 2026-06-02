"""Deterministic evaluation harness and pinned sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from weiss_rl.artifacts.reproducibility import (
    canonical_json_bytes,
    key256_to_hex,
    key256_to_short64,
    resolve_episode_key256,
    stable_hash64,
)
from weiss_rl.core.masking import (
    assert_strictly_increasing_legal_ids,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
)
from weiss_rl.core.termination_reason import classify_episode_end_reason
from weiss_rl.eval.engine_faults import abort_on_engine_fault_eval as _abort_on_engine_fault_eval
from weiss_rl.eval.policies.set import deck_id_for_policy_id
from weiss_rl.eval.sampling_helpers import coerce_eval_legal_ids as _coerce_eval_legal_ids
from weiss_rl.eval.sampling_helpers import coerce_eval_logits as _coerce_eval_logits
from weiss_rl.eval.sampling_helpers import coerce_sampling_temperature as _coerce_sampling_temperature
from weiss_rl.eval.sampling_helpers import legal_probs_for_cdf as _legal_probs_for_cdf
from weiss_rl.eval.sampling_helpers import normalize_cdf_probs
from weiss_rl.eval.sampling_helpers import require_pass_action as _require_pass_action
from weiss_rl.eval.sampling_helpers import sample_cdf_index as _sample_cdf_index
from weiss_rl.eval.sampling_helpers import selected_logp as _selected_logp
from weiss_rl.eval.terminal_step import MISSING as _MISSING
from weiss_rl.eval.terminal_step import optional_step_scalar as _optional_step_scalar
from weiss_rl.eval.terminal_step import require_seat as _require_seat
from weiss_rl.eval.terminal_step import required_step_scalar_with_fallback as _required_step_scalar_with_fallback
from weiss_rl.eval.terminal_step import step_scalar as _step_scalar
from weiss_rl.eval.terminal_step import winner_seat_from_terminal_step as _winner_seat_from_terminal_step

_U32_MASK = (1 << 32) - 1

OutcomeToken = Literal["W", "L", "D", "T"]


class _FloatRng(Protocol):
    def next_float(self) -> float: ...


class EvalGameRunner(Protocol):
    def run_game(self, scheduled_game: ScheduledGame) -> GameResult: ...


@dataclass(slots=True)
class EvalSamplerAnomalies:
    cdf_renormalizations: int = 0


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    pair_index: int
    swap_index: int
    episode_index: int
    episode_seed: int
    focal_policy_id: str
    opponent_policy_id: str
    seat0_policy_id: str
    seat1_policy_id: str
    focal_seat: int
    seat0_deck: str | None = None
    seat1_deck: str | None = None


@dataclass(frozen=True, slots=True)
class GameResult:
    episode_seed: int
    terminated: bool
    truncated: bool
    winner_seat: int | None
    engine_status: int = 0
    decision_count: int = 0
    tick_count: int = 0
    no_progress_count: int = 0
    termination_reason: str | None = None
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0
    simulator_episode_key: int | bytes | None = None
    replay_sample: ReplaySampleResult | None = None


@dataclass(frozen=True, slots=True)
class ReplaySampleResult:
    pair_index: int
    swap_index: int
    episode_index: int
    focal_policy_id: str
    opponent_policy_id: str
    raw_replay_path: str | None
    bundle_path: str
    verification_report_path: str
    verification_status: str
    replay_key64: str
    matched: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalGameRecord:
    pair_index: int
    swap_index: int
    episode_index: int
    episode_seed: int
    episode_key: str
    episode_key64: int
    config_hash256: str
    spec_hash256: str
    focal_policy_id: str
    opponent_policy_id: str
    seat0_policy_id: str
    seat1_policy_id: str
    focal_seat: int
    outcome: OutcomeToken
    terminated: bool
    truncated: bool
    engine_status: int
    seat0_deck: str | None = None
    seat1_deck: str | None = None
    decision_count: int = 0
    tick_count: int = 0
    no_progress_count: int = 0
    termination_reason: str = "terminated"
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0
    run_id256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_hash256": self.config_hash256,
            "engine_status": self.engine_status,
            "episode_index": self.episode_index,
            "episode_key": self.episode_key,
            "episode_key64": self.episode_key64,
            "episode_seed": self.episode_seed,
            "focal_policy_id": self.focal_policy_id,
            "focal_seat": self.focal_seat,
            "opponent_policy_id": self.opponent_policy_id,
            "outcome": self.outcome,
            "pair_index": self.pair_index,
            "seat0_policy_id": self.seat0_policy_id,
            "seat1_policy_id": self.seat1_policy_id,
            "spec_hash256": self.spec_hash256,
            "swap_index": self.swap_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "decision_count": self.decision_count,
            "tick_count": self.tick_count,
            "no_progress_count": self.no_progress_count,
            "termination_reason": self.termination_reason,
            "total_actions": self.total_actions,
            "pass_actions": self.pass_actions,
            "main_move_actions": self.main_move_actions,
            "pass_with_nonpass_available": self.pass_with_nonpass_available,
            "max_consecutive_main_moves": self.max_consecutive_main_moves,
        }
        if self.seat0_deck is not None:
            payload["seat0_deck"] = self.seat0_deck
        if self.seat1_deck is not None:
            payload["seat1_deck"] = self.seat1_deck
        if self.run_id256 is not None:
            payload["run_id256"] = self.run_id256
        return payload


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    episodes_path: Path
    records: tuple[EvalGameRecord, ...]
    summary: MatchupSummary


@dataclass(slots=True)
class MatchupSummary:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncations: int = 0
    engine_errors: int = 0
    natural_timeouts: int = 0
    no_progress_timeouts: int = 0
    decision_limit_timeouts: int = 0
    tick_limit_timeouts: int = 0
    timeout_unknown: int = 0
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0


def eval_sampler_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def eval_sampler_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def _normalize_cdf_probs(
    probs64: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    return normalize_cdf_probs(probs64, anomalies=anomalies)


def sample_action_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    rng: _FloatRng,
    pass_action_id: int | None = None,
    anomalies: EvalSamplerAnomalies | None = None,
    temperature: float = 1.0,
) -> tuple[int, np.float32]:
    """Sample one action from a single packed legal-id row with pinned CPU CDF math."""
    logits_array = _coerce_eval_logits(logits)
    temperature_value = _coerce_sampling_temperature(temperature)
    if temperature_value != 1.0:
        logits_array = logits_array / np.float32(temperature_value)
    legal_ids_array = _coerce_eval_legal_ids(legal_ids, action_space=logits_array.shape[0])

    if legal_ids_array.size == 0:
        action = _require_pass_action(pass_action_id, action_space=logits_array.shape[0])
        logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=action)
        return action, logp

    assert_strictly_increasing_legal_ids(legal_ids_array)
    probs64 = _legal_probs_for_cdf(logits_array, legal_ids_array, anomalies=anomalies)
    action_index = _sample_cdf_index(probs64, rng=rng)
    action = int(legal_ids_array[action_index])
    logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=pass_action_id)
    return action, logp


def select_action_argmax_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> tuple[int, np.float32]:
    """Select the highest-logit legal action with the same logp math as pinned eval."""

    logits_array = _coerce_eval_logits(logits)
    legal_ids_array = _coerce_eval_legal_ids(legal_ids, action_space=logits_array.shape[0])

    if legal_ids_array.size == 0:
        action = _require_pass_action(pass_action_id, action_space=logits_array.shape[0])
        logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=action)
        return action, logp

    assert_strictly_increasing_legal_ids(legal_ids_array)
    legal_logits = logits_array[legal_ids_array]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")
    action_index = int(np.argmax(legal_logits))
    action = int(legal_ids_array[action_index])
    logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=pass_action_id)
    return action, logp


def build_seat_swapped_schedule(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
) -> list[ScheduledGame]:
    schedule: list[ScheduledGame] = []
    for pair_index, raw_seed in enumerate(paired_seeds):
        episode_seed = int(raw_seed)
        schedule.append(
            ScheduledGame(
                pair_index=pair_index,
                swap_index=0,
                episode_index=len(schedule),
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=focal_policy_id,
                seat1_policy_id=opponent_policy_id,
                focal_seat=0,
                seat0_deck=deck_id_for_policy_id(focal_policy_id),
                seat1_deck=deck_id_for_policy_id(opponent_policy_id),
            )
        )
        schedule.append(
            ScheduledGame(
                pair_index=pair_index,
                swap_index=1,
                episode_index=len(schedule),
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=opponent_policy_id,
                seat1_policy_id=focal_policy_id,
                focal_seat=1,
                seat0_deck=deck_id_for_policy_id(opponent_policy_id),
                seat1_deck=deck_id_for_policy_id(focal_policy_id),
            )
        )
    return schedule


def run_seat_swapped_matchup(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
    runner: EvalGameRunner,
    episodes_path: Path,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
) -> EvalRunResult:
    schedule = build_seat_swapped_schedule(
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seeds=paired_seeds,
    )
    records = [
        record_completed_game(
            scheduled_game=game,
            result=runner.run_game(game),
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
        )
        for game in schedule
    ]
    write_episodes_jsonl(episodes_path, records)
    return EvalRunResult(
        episodes_path=episodes_path,
        records=tuple(records),
        summary=summarize_game_records(records),
    )


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


def _validate_completed_game_result(result: GameResult) -> None:
    terminated = bool(result.terminated)
    truncated = bool(result.truncated)
    if terminated == truncated:
        raise ValueError("completed game result must set exactly one of terminated or truncated")
    if truncated and result.winner_seat is not None:
        raise ValueError("truncated game result cannot include winner_seat")


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


def abort_on_engine_fault_eval(
    *,
    run_dir: Path,
    engine_status: Any,
    decision_id: Any | None = None,
    episode_key: Any | None = None,
    note: str = "engine_status!=0 during evaluation",
) -> None:
    _abort_on_engine_fault_eval(
        run_dir=run_dir,
        engine_status=engine_status,
        decision_id=decision_id,
        episode_key=episode_key,
        note=note,
    )


def game_result_from_step(
    step: object,
    *,
    env_index: int = 0,
    acting_seat: int | None = None,
    episode_seed: int | None = None,
    max_decisions: int | None = None,
    max_ticks: int | None = None,
    max_no_progress_decisions: int | None = None,
) -> GameResult:
    """Decode one environment row into an evaluation result.

    Prefer explicit terminal winner metadata when the step exposes it. Otherwise
    decisive terminated rows are inferred from reward sign relative to the
    acting seat, and a terminated zero reward is treated as a draw fallback.
    That zero-reward draw fallback matches the locked thesis configs and should
    be revisited if terminal shaping semantics change.

    Some minimal terminal step objects omit context fields such as acting seat
    or episode seed; callers may supply those explicitly when unavailable on
    the observed row.
    """
    reward = _step_scalar(step, ("reward", "rewards"), env_index=env_index, cast_fn=float)
    terminated = _step_scalar(step, ("terminated",), env_index=env_index, cast_fn=bool)
    truncated = _step_scalar(step, ("truncated",), env_index=env_index, cast_fn=bool)
    engine_status = _step_scalar(step, ("engine_status",), env_index=env_index, cast_fn=int)
    decision_count = _optional_step_scalar(step, ("decision_count",), env_index=env_index)
    tick_count = _optional_step_scalar(step, ("tick_count",), env_index=env_index)
    no_progress_count = _optional_step_scalar(step, ("no_progress_count",), env_index=env_index)
    decision_count_i = 0 if decision_count is None else int(decision_count)
    tick_count_i = 0 if tick_count is None else int(tick_count)
    no_progress_count_i = 0 if no_progress_count is None else int(no_progress_count)
    resolved_episode_seed = _required_step_scalar_with_fallback(
        step,
        ("episode_seed",),
        env_index=env_index,
        cast_fn=int,
        fallback=_MISSING if episode_seed is None else episode_seed,
        fallback_name="episode_seed",
    )
    simulator_episode_key = _optional_step_scalar(step, ("episode_key",), env_index=env_index)

    winner_seat = _winner_seat_from_terminal_step(
        step,
        env_index=env_index,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        acting_seat=acting_seat,
    )
    termination_reason = classify_episode_end_reason(
        terminated=terminated,
        truncated=truncated,
        engine_status=engine_status,
        decision_count=decision_count_i,
        tick_count=tick_count_i,
        no_progress_count=no_progress_count_i,
        max_decisions=max_decisions,
        max_ticks=max_ticks,
        max_no_progress_decisions=max_no_progress_decisions,
    )

    return GameResult(
        episode_seed=resolved_episode_seed,
        terminated=terminated,
        truncated=truncated,
        winner_seat=winner_seat,
        engine_status=engine_status,
        decision_count=decision_count_i,
        tick_count=tick_count_i,
        no_progress_count=no_progress_count_i,
        termination_reason=termination_reason,
        simulator_episode_key=simulator_episode_key,
    )


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
