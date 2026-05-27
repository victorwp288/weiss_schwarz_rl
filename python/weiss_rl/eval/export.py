"""Summary export helpers for seat-swapped evaluation records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.harness import EvalGameRecord, OutcomeToken
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.stage2 import summarize_stage2_records

__all__ = [
    "build_matchup_export",
    "load_eval_game_records",
    "write_matchup_summary_csv",
    "write_matchup_summary_json",
]


def load_eval_game_records(path: Path) -> tuple[EvalGameRecord, ...]:
    records: list[EvalGameRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object payload on line {line_number}")
            try:
                outcome = str(payload["outcome"])
                if outcome not in {"W", "L", "D", "T"}:
                    raise ValueError(f"invalid outcome {outcome!r} on line {line_number}")
                raw_run_id256 = payload.get("run_id256")
                record = EvalGameRecord(
                    pair_index=int(payload["pair_index"]),
                    swap_index=int(payload["swap_index"]),
                    episode_index=int(payload["episode_index"]),
                    episode_seed=int(payload["episode_seed"]),
                    episode_key=str(payload["episode_key"]),
                    episode_key64=int(payload["episode_key64"]),
                    config_hash256=str(payload["config_hash256"]),
                    spec_hash256=str(payload["spec_hash256"]),
                    focal_policy_id=str(payload["focal_policy_id"]),
                    opponent_policy_id=str(payload["opponent_policy_id"]),
                    seat0_policy_id=str(payload["seat0_policy_id"]),
                    seat1_policy_id=str(payload["seat1_policy_id"]),
                    focal_seat=int(payload["focal_seat"]),
                    outcome=cast(OutcomeToken, outcome),
                    terminated=bool(payload["terminated"]),
                    truncated=bool(payload["truncated"]),
                    engine_status=int(payload["engine_status"]),
                    seat0_deck=None if payload.get("seat0_deck") is None else str(payload["seat0_deck"]),
                    seat1_deck=None if payload.get("seat1_deck") is None else str(payload["seat1_deck"]),
                    decision_count=int(payload.get("decision_count", 0)),
                    tick_count=int(payload.get("tick_count", 0)),
                    no_progress_count=int(payload.get("no_progress_count", 0)),
                    termination_reason=str(payload.get("termination_reason", "terminated")),
                    total_actions=int(payload.get("total_actions", 0)),
                    pass_actions=int(payload.get("pass_actions", 0)),
                    main_move_actions=int(payload.get("main_move_actions", 0)),
                    pass_with_nonpass_available=int(payload.get("pass_with_nonpass_available", 0)),
                    max_consecutive_main_moves=int(payload.get("max_consecutive_main_moves", 0)),
                    run_id256=_normalize_optional_hash256(raw_run_id256, name="run_id256"),
                )
            except KeyError as exc:
                raise ValueError(f"missing required field {exc.args[0]!r} on line {line_number}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"no EvalGameRecord rows found in {path}")
    return tuple(records)


def build_matchup_export(
    records: tuple[EvalGameRecord, ...] | list[EvalGameRecord],
    *,
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
    scheme: PayoffFoldScheme = "S0",
    sample_count: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    focal_policy_id, opponent_policy_id = _matchup_ids(records)
    _require_single_contract(records)
    deck_context = _matchup_deck_context(records)
    decision = summarize_stage2_records(
        records,
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
        scheme=scheme,
        sample_count=sample_count,
        seed=seed,
    )
    summary = decision.summary
    uncertainty = decision.uncertainty
    uncertainty_payload: dict[str, Any] = {
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "ci_half_width": None,
        "prob_gt_half": None,
        "prob_lt_half": None,
        "paired_seed_count": decision.paired_seed_count,
        "sample_count": 0,
    }
    if uncertainty is not None:
        uncertainty_payload = {
            "mean": uncertainty.mean,
            "ci_low": uncertainty.ci_low,
            "ci_high": uncertainty.ci_high,
            "ci_half_width": uncertainty.ci_half_width,
            "prob_gt_half": uncertainty.prob_gt_half,
            "prob_lt_half": uncertainty.prob_lt_half,
            "paired_seed_count": uncertainty.paired_seed_count,
            "sample_count": uncertainty.sample_count,
        }
    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "deck_context": deck_context,
        "scheme": scheme,
        "paired_seeds": decision.paired_seed_count,
        "observed_paired_seeds": decision.observed_paired_seeds,
        "excluded_paired_seeds": decision.excluded_paired_seeds,
        "has_payoff_samples": decision.has_payoff_samples,
        "max_paired_seeds": decision.max_paired_seeds,
        "stop_reason": decision.stop_reason,
        "should_stop": decision.should_stop,
        "summary": {
            "games": summary.games,
            "wins": summary.wins,
            "losses": summary.losses,
            "draws": summary.draws,
            "truncations": summary.truncations,
            "engine_errors": summary.engine_errors,
            "natural_timeouts": summary.natural_timeouts,
            "no_progress_timeouts": summary.no_progress_timeouts,
            "decision_limit_timeouts": summary.decision_limit_timeouts,
            "tick_limit_timeouts": summary.tick_limit_timeouts,
            "timeout_unknown": summary.timeout_unknown,
            "total_actions": summary.total_actions,
            "pass_actions": summary.pass_actions,
            "main_move_actions": summary.main_move_actions,
            "pass_with_nonpass_available": summary.pass_with_nonpass_available,
            "max_consecutive_main_moves": summary.max_consecutive_main_moves,
        },
        "uncertainty": uncertainty_payload,
    }


def write_matchup_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_matchup_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    uncertainty = payload["uncertainty"]
    row = {
        "focal_policy_id": payload["focal_policy_id"],
        "opponent_policy_id": payload["opponent_policy_id"],
        "focal_deck": payload.get("deck_context", {}).get("focal_deck"),
        "opponent_deck": payload.get("deck_context", {}).get("opponent_deck"),
        "scheme": payload["scheme"],
        "paired_seeds": payload["paired_seeds"],
        "max_paired_seeds": payload["max_paired_seeds"],
        "observed_paired_seeds": payload["observed_paired_seeds"],
        "excluded_paired_seeds": payload["excluded_paired_seeds"],
        "has_payoff_samples": payload["has_payoff_samples"],
        "stop_reason": payload["stop_reason"],
        "should_stop": payload["should_stop"],
        "games": summary["games"],
        "wins": summary["wins"],
        "losses": summary["losses"],
        "draws": summary["draws"],
        "truncations": summary["truncations"],
        "engine_errors": summary["engine_errors"],
        "natural_timeouts": summary["natural_timeouts"],
        "no_progress_timeouts": summary["no_progress_timeouts"],
        "decision_limit_timeouts": summary["decision_limit_timeouts"],
        "tick_limit_timeouts": summary["tick_limit_timeouts"],
        "timeout_unknown": summary["timeout_unknown"],
        "total_actions": summary["total_actions"],
        "pass_actions": summary["pass_actions"],
        "main_move_actions": summary["main_move_actions"],
        "pass_with_nonpass_available": summary["pass_with_nonpass_available"],
        "max_consecutive_main_moves": summary["max_consecutive_main_moves"],
        "mean": uncertainty["mean"],
        "ci_low": uncertainty["ci_low"],
        "ci_high": uncertainty["ci_high"],
        "ci_half_width": uncertainty["ci_half_width"],
        "prob_gt_half": uncertainty["prob_gt_half"],
        "prob_lt_half": uncertainty["prob_lt_half"],
        "sample_count": uncertainty["sample_count"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _matchup_ids(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> tuple[str, str]:
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError("summary export expects records for exactly one focal/opponent matchup")
    return next(iter(focal_ids)), next(iter(opponent_ids))


def _require_single_contract(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> None:
    config_hashes = {record.config_hash256 for record in records}
    spec_hashes = {record.spec_hash256 for record in records}
    if len(config_hashes) != 1 or len(spec_hashes) != 1:
        raise ValueError("summary export expects records for exactly one config/spec contract")


def _matchup_deck_context(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> dict[str, str | None]:
    focal_decks: set[str | None] = set()
    opponent_decks: set[str | None] = set()
    for record in records:
        if record.focal_seat == 0:
            focal_decks.add(record.seat0_deck)
            opponent_decks.add(record.seat1_deck)
        else:
            focal_decks.add(record.seat1_deck)
            opponent_decks.add(record.seat0_deck)
    return {
        "focal_deck": _single_deck_or_none(focal_decks),
        "opponent_deck": _single_deck_or_none(opponent_decks),
    }


def _single_deck_or_none(decks: set[str | None]) -> str | None:
    normalized = {deck for deck in decks if deck is not None}
    if len(normalized) == 1:
        return next(iter(normalized))
    return None


def _normalize_optional_hash256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be 64 hex chars, got {len(normalized)}")
    bytes.fromhex(normalized)
    return normalized
