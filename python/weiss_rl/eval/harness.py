"""Deterministic evaluation harness and pinned sampling helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from weiss_rl.masking import assert_strictly_increasing_legal_ids, masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.repro import canonical_json_bytes, key256_to_hex, key256_to_short64, resolve_episode_key256, stable_hash64

_CDF_RENORMALIZE_TOL = 1e-6
_U32_MASK = (1 << 32) - 1
_MISSING = object()

OutcomeToken = Literal["W", "L", "D", "T"]


class _FloatRng(Protocol):
    def next_float(self) -> float: ...


class EvalGameRunner(Protocol):
    def run_game(self, scheduled_game: "ScheduledGame") -> "GameResult": ...


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


@dataclass(frozen=True, slots=True)
class GameResult:
    episode_seed: int
    terminated: bool
    truncated: bool
    winner_seat: int | None
    engine_status: int = 0
    simulator_episode_key: int | bytes | None = None


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    episodes_path: Path
    records: tuple[EvalGameRecord, ...]
    summary: "MatchupSummary"


@dataclass(slots=True)
class MatchupSummary:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncations: int = 0
    engine_errors: int = 0


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


def sample_action_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    rng: _FloatRng,
    pass_action_id: int | None = None,
    anomalies: EvalSamplerAnomalies | None = None,
) -> tuple[int, np.float32]:
    """Sample one action from a single packed legal-id row with pinned CPU CDF math."""
    logits_array = _coerce_eval_logits(logits)
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
    summary.engine_errors = sum(1 for record in records if record.engine_status != 0)
    return summary


def write_episodes_jsonl(path: Path, records: Sequence[EvalGameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record.to_dict()).decode("utf-8"))
            handle.write("\n")


def _fault_env_indices(engine_status: Any) -> list[int]:
    return np.flatnonzero(np.atleast_1d(np.asarray(engine_status)) != 0).astype(int).tolist()


def _json_ready_array(value: Any) -> int | list[int]:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array)
    return array.astype(int).tolist()


def _json_ready_episode_key(episode_key: Any) -> object:
    if isinstance(episode_key, (bytes, bytearray)):
        return repr(bytes(episode_key))

    array = np.asarray(episode_key)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (bytes, bytearray)):
            return repr(bytes(scalar))
        return scalar
    return array.tolist()


def abort_on_engine_fault_eval(
    *,
    run_dir: Path,
    engine_status: Any,
    decision_id: Any | None = None,
    episode_key: Any | None = None,
    note: str = "engine_status!=0 during evaluation",
) -> None:
    """Hard-fail evaluation on engine faults after writing a local artifact."""
    fault_env_indices = _fault_env_indices(engine_status)
    if not fault_env_indices:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    fault_path = run_dir / "eval_engine_fault.json"
    payload: dict[str, object] = {
        "note": note,
        "fault_env_indices": fault_env_indices,
        "engine_status": _json_ready_array(engine_status),
    }
    if decision_id is not None:
        payload["decision_id"] = _json_ready_array(decision_id)
    if episode_key is not None:
        payload["episode_key"] = _json_ready_episode_key(episode_key)

    fault_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    raise RuntimeError(f"{note}; wrote {fault_path}")


def game_result_from_step(
    step: object,
    *,
    env_index: int = 0,
    acting_seat: int | None = None,
    episode_seed: int | None = None,
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

    return GameResult(
        episode_seed=resolved_episode_seed,
        terminated=terminated,
        truncated=truncated,
        winner_seat=winner_seat,
        engine_status=engine_status,
        simulator_episode_key=simulator_episode_key,
    )


def _winner_seat_from_terminal_step(
    step: object,
    *,
    env_index: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    acting_seat: int | None,
) -> int | None:
    if not terminated or truncated:
        return None

    explicit_winner_seat = _optional_terminal_winner_seat(step, env_index=env_index)
    if explicit_winner_seat is not _MISSING:
        return cast(int | None, explicit_winner_seat)
    if reward == 0.0:
        return None

    perspective_seat = _reward_perspective_seat(step, env_index=env_index, acting_seat=acting_seat)
    return perspective_seat if reward > 0.0 else 1 - perspective_seat


def _optional_terminal_winner_seat(step: object, *, env_index: int) -> object:
    for name in ("winner_seat", "winner"):
        if not hasattr(step, name):
            continue
        value = _step_value_for_env(step, name=name, env_index=env_index)
        if value is None:
            return None

        seat = int(value)
        if seat == -1:
            return None
        return _require_seat(seat, name=name)
    return _MISSING


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
    matchup_id = f"{scheduled_game.focal_policy_id}\0{scheduled_game.opponent_policy_id}"
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


def _require_seat(value: int, *, name: str) -> int:
    seat = int(value)
    if seat not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {seat}")
    return seat


def _step_value_for_env(step: object, *, name: str, env_index: int) -> Any:
    values = np.asarray(getattr(step, name), dtype=object)
    if values.ndim == 0:
        value = values.item()
    else:
        value = values[env_index]
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _step_scalar(
    step: object,
    names: Sequence[str],
    *,
    env_index: int,
    cast_fn: Any,
) -> Any:
    for name in names:
        if hasattr(step, name):
            return cast_fn(_step_value_for_env(step, name=name, env_index=env_index))
    joined_names = ", ".join(names)
    raise AttributeError(f"step is missing required field(s): {joined_names}")


def _required_step_scalar_with_fallback(
    step: object,
    names: Sequence[str],
    *,
    env_index: int,
    cast_fn: Any,
    fallback: Any,
    fallback_name: str,
) -> Any:
    try:
        observed = _step_scalar(step, names, env_index=env_index, cast_fn=cast_fn)
    except AttributeError:
        if fallback is _MISSING:
            joined_names = ", ".join(names)
            raise AttributeError(
                f"step is missing required field(s): {joined_names}; provide {fallback_name} when unavailable"
            ) from None
        return cast_fn(fallback)

    if fallback is _MISSING:
        return observed

    expected = cast_fn(fallback)
    if observed != expected:
        raise ValueError(f"{fallback_name} mismatch: step={observed}, provided={expected}")
    return observed


def _optional_step_scalar(step: object, names: Sequence[str], *, env_index: int) -> int | bytes | None:
    for name in names:
        if not hasattr(step, name):
            continue
        value = _step_value_for_env(step, name=name, env_index=env_index)
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return int(value)
    return None


def _reward_perspective_seat(step: object, *, env_index: int, acting_seat: int | None) -> int:
    aliases: list[tuple[str, int]] = []
    if acting_seat is not None:
        aliases.append(("acting_seat", _require_seat(acting_seat, name="acting_seat")))

    for name in ("actor", "to_play_seat", "to_play"):
        if not hasattr(step, name):
            continue
        seat = _step_scalar(step, (name,), env_index=env_index, cast_fn=int)
        if seat == -1:
            continue
        aliases.append((name, _require_seat(seat, name=name)))

    if not aliases:
        raise AttributeError(
            "decisive terminated step must expose acting_seat or a valid actor, to_play_seat, or to_play"
        )

    canonical_name, canonical_seat = aliases[0]
    for name, seat in aliases[1:]:
        if seat != canonical_seat:
            raise ValueError(f"reward perspective seat mismatch: {canonical_name}={canonical_seat}, {name}={seat}")
    return canonical_seat


def _coerce_eval_logits(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 1:
        raise ValueError("logits must be a 1D array")
    return logits_array


def _coerce_eval_legal_ids(legal_ids: np.ndarray, *, action_space: int) -> np.ndarray:
    legal_ids_array = np.asarray(legal_ids)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.dtype == np.bool_ or not np.issubdtype(legal_ids_array.dtype, np.integer):
        raise ValueError("legal_ids must be an integer array")

    signed = legal_ids_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("legal_ids must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"legal_ids must be < action_space ({action_space})")
    return signed.astype(np.intp, copy=False)


def _require_pass_action(pass_action_id: int | None, *, action_space: int) -> int:
    if pass_action_id is None:
        raise ValueError("pass_action_id is required when legal_ids is empty")
    if pass_action_id < 0 or pass_action_id >= action_space:
        raise ValueError(f"pass_action_id must be in [0, {action_space})")
    return int(pass_action_id)


def _legal_probs_for_cdf(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    legal_logits = logits[legal_ids]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")

    row_max = np.max(legal_logits)
    shifted = legal_logits - row_max
    weights = np.exp(shifted)
    denom = np.sum(weights, dtype=np.float32)
    probs64 = np.asarray(weights / denom, dtype=np.float64)
    return _normalize_cdf_probs(probs64, anomalies=anomalies)


def _normalize_cdf_probs(
    probs64: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    prob_sum = float(np.sum(probs64, dtype=np.float64))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("legal probabilities must sum to a finite positive value")
    if abs(prob_sum - 1.0) > _CDF_RENORMALIZE_TOL:
        probs64 = probs64 / prob_sum
        if anomalies is not None:
            anomalies.cdf_renormalizations += 1
    return probs64


def _sample_cdf_index(probs64: np.ndarray, *, rng: _FloatRng) -> int:
    cdf = np.cumsum(probs64, dtype=np.float64)
    cdf[-1] = 1.0
    draw = float(rng.next_float())
    if not np.isfinite(draw) or draw < 0.0 or draw > 1.0:
        raise ValueError("rng.next_float() must return a finite value in [0.0, 1.0]")
    return min(int(np.searchsorted(cdf, draw, side="right")), cdf.size - 1)


def _selected_logp(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    action: int,
    *,
    pass_action_id: int | None,
) -> np.float32:
    legal_offsets = np.array([0, legal_ids.size], dtype=np.int64)
    actions = np.array([action], dtype=np.int64)
    logp = masked_logp_from_legal_ids(
        logits[np.newaxis, :],
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )
    return np.float32(logp[0])
