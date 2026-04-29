"""Run a small B1 artifact matrix with load manifests and pair tables.

This is a diagnostic script, not a thesis comparison surface. It is meant to
answer whether B1 parity could be caused by stale loading, alias resolution, or
paired-seat summarization before spending more compute on learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import train as train_script
from weiss_rl.action_catalog import ActionCatalog, DecodedAction
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.diagnostics import build_seat_advantage_diagnostics
from weiss_rl.eval.export import build_matchup_export, write_matchup_summary_json
from weiss_rl.eval.harness import EvalGameRecord, ScheduledGame, run_seat_swapped_matchup, sample_action_pinned
from weiss_rl.eval.payoff_folding import paired_seed_score, validated_paired_seed_groups
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)
from weiss_rl.eval.simulator_runner import (
    NO_LEAGUE_POLICY_ID,
    ResolvedEvalPolicy,
    SimulatorEvalRunner,
    resolve_eval_policies,
)
from weiss_rl.repro import canonical_json_bytes, stable_hash64


def _manifest_value(manifest: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_")
    return slug or "policy"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(array.shape)).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _state_dict_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for tensor in model.state_dict().values():
        if not torch.is_floating_point(tensor):
            continue
        value = tensor.detach().cpu().double()
        total += float(torch.sum(value * value).item())
    return float(math.sqrt(total))


def _state_dict_l2_distance(left: torch.nn.Module, right: torch.nn.Module) -> dict[str, Any]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    shared = sorted(set(left_state) & set(right_state))
    total = 0.0
    compared = 0
    skipped: list[str] = []
    for name in shared:
        left_tensor = left_state[name]
        right_tensor = right_state[name]
        if left_tensor.shape != right_tensor.shape or not torch.is_floating_point(left_tensor):
            skipped.append(name)
            continue
        delta = left_tensor.detach().cpu().double() - right_tensor.detach().cpu().double()
        total += float(torch.sum(delta * delta).item())
        compared += int(delta.numel())
    return {
        "l2_distance": float(math.sqrt(total)),
        "compared_float_params": compared,
        "shared_keys": len(shared),
        "skipped_keys": skipped[:32],
        "skipped_key_count": len(skipped),
    }


def _policy_manifest_entry(
    *,
    policy_id: str,
    policy: ResolvedEvalPolicy,
    source_path: Path | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "policy_id": policy_id,
        "resolved_policy_id": policy.policy_id,
        "kind": policy.kind,
        "source_run_dir": policy.source_run_dir,
        "snapshot_path": policy.snapshot_path,
        "source_path": None if source_path is None else source_path.as_posix(),
        "source_file_sha256": None if source_path is None or not source_path.is_file() else _sha256_file(source_path),
        "model_object_id": None if policy.model is None else id(policy.model),
        "state_dict_sha256": None if policy.model is None else _state_dict_sha256(policy.model),
        "state_dict_param_l2_norm": None if policy.model is None else _state_dict_l2_norm(policy.model),
    }
    if policy.model is not None:
        get_bias_scale = getattr(policy.model, "get_public_heuristic_logit_bias_scale", None)
        if callable(get_bias_scale):
            try:
                entry["public_heuristic_logit_bias_scale_learner"] = float(get_bias_scale(scoring_mode="learner"))
                entry["public_heuristic_logit_bias_scale_actor"] = float(get_bias_scale(scoring_mode="actor"))
            except Exception as exc:  # pragma: no cover - defensive diagnostic payload
                entry["public_heuristic_logit_bias_scale_error"] = str(exc)
    return entry


def _winner_seat(record: EvalGameRecord) -> int | None:
    if record.outcome == "W":
        return int(record.focal_seat)
    if record.outcome == "L":
        return 1 - int(record.focal_seat)
    return None


def _record_payload(record: EvalGameRecord) -> dict[str, Any]:
    payload = {
        "pair_index": int(record.pair_index),
        "swap_index": int(record.swap_index),
        "episode_seed": int(record.episode_seed),
        "episode_key64": int(record.episode_key64),
        "focal_seat": int(record.focal_seat),
        "outcome": record.outcome,
        "winner_seat": _winner_seat(record),
        "seat0_policy_id": record.seat0_policy_id,
        "seat1_policy_id": record.seat1_policy_id,
        "terminated": bool(record.terminated),
        "truncated": bool(record.truncated),
        "engine_status": int(record.engine_status),
        "decision_count": int(record.decision_count),
        "tick_count": int(record.tick_count),
        "termination_reason": record.termination_reason,
    }
    if record.terminal_summary is not None:
        payload["terminal_summary"] = dict(record.terminal_summary)
    return payload


def _pair_table(records: Sequence[EvalGameRecord], *, scheme: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {"2-0": 0, "1-1": 0, "0-2": 0, "mixed": 0}
    for pair_records in validated_paired_seed_groups(records):
        by_swap = {int(record.swap_index): record for record in pair_records}
        swap0 = by_swap[0]
        swap1 = by_swap[1]
        outcomes = (swap0.outcome, swap1.outcome)
        if outcomes == ("W", "W"):
            pair_class = "2-0"
        elif outcomes == ("L", "L"):
            pair_class = "0-2"
        elif set(outcomes) == {"W", "L"}:
            pair_class = "1-1"
        else:
            pair_class = "mixed"
        counts[pair_class] += 1
        rows.append(
            {
                "pair_index": int(swap0.pair_index),
                "episode_seed": int(swap0.episode_seed),
                "focal_policy_id": swap0.focal_policy_id,
                "opponent_policy_id": swap0.opponent_policy_id,
                "pair_score": paired_seed_score(pair_records, scheme=scheme),
                "pair_class": pair_class,
                "focal_as_seat0": _record_payload(swap0),
                "focal_as_seat1": _record_payload(swap1),
            }
        )
    summary = {
        "pair_count": len(rows),
        "pair_class_counts": counts,
        "pair_score_mean": float(np.mean([row["pair_score"] for row in rows if row["pair_score"] is not None]))
        if rows
        else None,
    }
    return rows, summary


def _complement_check(
    matrix: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, Any] | None:
    forward = matrix.get((left, right))
    reverse = matrix.get((right, left))
    if forward is None or reverse is None:
        return None
    forward_mean = float(forward["matchup_payload"]["uncertainty"]["mean"])
    reverse_mean = float(reverse["matchup_payload"]["uncertainty"]["mean"])
    return {
        "left": left,
        "right": right,
        "forward_mean": forward_mean,
        "reverse_mean": reverse_mean,
        "sum": forward_mean + reverse_mean,
        "abs_sum_minus_one": abs((forward_mean + reverse_mean) - 1.0),
    }


def _parse_policy_arg(raw: str) -> tuple[str, Path]:
    alias, sep, path_text = str(raw).partition("=")
    if not sep or not alias.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("--checkpoint-policy values must be alias=path")
    return alias.strip(), Path(path_text.strip())


def _format_decoded_action(action: DecodedAction) -> str:
    if action.family in {"pass", "mulligan_confirm", "choice_prev_page", "choice_next_page", "concede"}:
        return action.family
    if action.family in {"clock_from_hand", "main_play_event", "climax_play"}:
        return _format_with_fields(action.family, ("hand_index", action.hand_index))
    if action.family in {"level_up", "trigger_order", "choice_select", "mulligan_select"}:
        return _format_with_fields(action.family, ("index", action.index), ("hand_index", action.hand_index))
    if action.family == "main_play_character":
        return _format_with_fields(action.family, ("hand_index", action.hand_index), ("stage_slot", action.stage_slot))
    if action.family == "main_move":
        return _format_with_fields(action.family, ("from_slot", action.from_slot), ("to_slot", action.to_slot))
    if action.family == "attack":
        return _format_with_fields(action.family, ("slot", action.slot), ("attack_type", action.attack_type))
    if action.family in {"encore_pay", "encore_decline"}:
        return _format_with_fields(action.family, ("slot", action.slot))
    return action.family


def _format_with_fields(family: str, *fields: tuple[str, Any]) -> str:
    payload = ", ".join(f"{name}={value}" for name, value in fields if value is not None)
    return family if not payload else f"{family}({payload})"


def _action_payload(action_id: int, *, action_catalog: ActionCatalog | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action_id": int(action_id)}
    if action_catalog is None:
        return payload
    try:
        decoded = action_catalog.decode(int(action_id))
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        payload["decode_error"] = str(exc)
        return payload
    payload["family"] = decoded.family
    payload["label"] = _format_decoded_action(decoded)
    return payload


def _topk_payload(
    *,
    logits: np.ndarray,
    legal_ids: np.ndarray,
    top_k: int,
    action_catalog: ActionCatalog | None,
) -> list[dict[str, Any]]:
    legal = np.asarray(legal_ids, dtype=np.int64)
    if legal.size == 0:
        return []
    legal_logits = logits[legal]
    order = np.argsort(legal_logits)[::-1][: max(0, int(top_k))]
    return [
        {
            **_action_payload(int(legal[int(index)]), action_catalog=action_catalog),
            "logit": float(legal_logits[int(index)]),
        }
        for index in order.tolist()
    ]


def _parse_matchup_arg(raw: str) -> tuple[str, str]:
    left, sep, right = str(raw).partition("=")
    if not sep or not left.strip() or not right.strip():
        raise argparse.ArgumentTypeError("--matchup values must be focal=opponent")
    return left.strip(), right.strip()


def _parse_force_action_sequence(raw: str) -> list[dict[str, int]]:
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--force-action-sequence must be JSON: {exc}") from exc
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise argparse.ArgumentTypeError("--force-action-sequence must be a JSON object or list")
    sequence: list[dict[str, int]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise argparse.ArgumentTypeError(f"force sequence item {index} must be an object")
        try:
            seat = int(item["seat"])
            decision_index = int(item["decision_index"])
            action_id = int(item["action_id"])
        except KeyError as exc:
            raise argparse.ArgumentTypeError(
                "force sequence items require seat, decision_index, and action_id"
            ) from exc
        swap_index = item.get("swap_index")
        if seat not in (0, 1):
            raise argparse.ArgumentTypeError(f"force sequence item {index} seat must be 0 or 1")
        if decision_index < 0 or action_id < 0:
            raise argparse.ArgumentTypeError(f"force sequence item {index} decision/action must be non-negative")
        normalized = {
            "seat": seat,
            "decision_index": decision_index,
            "action_id": action_id,
        }
        if swap_index is not None:
            normalized["swap_index"] = int(swap_index)
        sequence.append(normalized)
    return sequence


_BUILTIN_POLICY_ALIASES = {
    "B0": RANDOM_LEGAL_POLICY_ID,
    "B0_RANDOMLEGAL": RANDOM_LEGAL_POLICY_ID,
    "B0_RANDOM_LEGAL": RANDOM_LEGAL_POLICY_ID,
    "RANDOM": RANDOM_LEGAL_POLICY_ID,
    "RANDOMLEGAL": RANDOM_LEGAL_POLICY_ID,
    "B1": NO_LEAGUE_POLICY_ID,
    "B1_NOLEAGUE": NO_LEAGUE_POLICY_ID,
    "B1_NOLEAGUE_BASELINE": NO_LEAGUE_POLICY_ID,
    "B2": HEURISTIC_PUBLIC_POLICY_ID,
    "B2_HEURISTICPUBLIC": HEURISTIC_PUBLIC_POLICY_ID,
    "B2_HEURISTIC_PUBLIC": HEURISTIC_PUBLIC_POLICY_ID,
    "B3": HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    "B3_HEURISTICAGGRO": HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    "B3_HEURISTIC_AGGRO": HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    "B3_HEURISTICPUBLICAGGRO": HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    "B3_HEURISTIC_PUBLIC_AGGRO": HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    "B4": HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    "B4_HEURISTICCONTROL": HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    "B4_HEURISTIC_CONTROL": HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    "B4_HEURISTICPUBLICCONTROL": HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    "B4_HEURISTIC_PUBLIC_CONTROL": HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
}


def _canonical_builtin_policy_id(raw: str) -> str:
    stripped = str(raw).strip()
    key = re.sub(r"[^A-Za-z0-9]+", "_", stripped).strip("_").upper()
    policy_id = _BUILTIN_POLICY_ALIASES.get(key)
    if policy_id is None:
        allowed = ", ".join(
            sorted({"B0_RandomLegal", "B2_HeuristicPublic", "B3_HeuristicAggro", "B4_HeuristicControl"})
        )
        raise argparse.ArgumentTypeError(f"unknown builtin policy {raw!r}; expected one of: {allowed}")
    return policy_id


def _resolve_policy_alias(raw: str, policy_ids: Sequence[str]) -> str:
    if raw in policy_ids:
        return raw
    try:
        builtin = _canonical_builtin_policy_id(raw)
    except argparse.ArgumentTypeError:
        builtin = ""
    if builtin and builtin in policy_ids:
        return builtin
    slug_map = {_safe_slug(policy_id).lower(): policy_id for policy_id in policy_ids}
    slugged = _safe_slug(raw).lower()
    if slugged in slug_map:
        return slug_map[slugged]
    raise SystemExit(f"unknown matchup policy {raw!r}; available: {', '.join(policy_ids)}")


def _read_bias_scale(policy: ResolvedEvalPolicy, *, scoring_mode: str) -> float | None:
    if policy.model is None:
        return None
    get_bias_scale = getattr(policy.model, "get_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale):
        return None
    return float(get_bias_scale(scoring_mode=scoring_mode))


def _apply_public_heuristic_bias_override(
    policies: Mapping[str, ResolvedEvalPolicy],
    *,
    override_scale: float | None,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    if override_scale is None:
        for policy_id, policy in policies.items():
            report[policy_id] = {
                "requested": False,
                "effective_learner": _read_bias_scale(policy, scoring_mode="learner"),
                "effective_actor": _read_bias_scale(policy, scoring_mode="actor"),
            }
        return report
    expected = float(override_scale)
    for policy_id, policy in policies.items():
        before = {
            "learner": _read_bias_scale(policy, scoring_mode="learner"),
            "actor": _read_bias_scale(policy, scoring_mode="actor"),
        }
        if policy.model is not None:
            set_bias_scale = getattr(policy.model, "set_public_heuristic_logit_bias_scale", None)
            if not callable(set_bias_scale):
                raise RuntimeError(f"bias override requested, but policy {policy_id!r} has no bias setter")
            set_bias_scale(expected, actor_value=expected)
        after = {
            "learner": _read_bias_scale(policy, scoring_mode="learner"),
            "actor": _read_bias_scale(policy, scoring_mode="actor"),
        }
        if policy.model is not None:
            for mode, value in after.items():
                if value is None or abs(float(value) - expected) > 1e-6:
                    raise RuntimeError(
                        f"bias override requested, but policy {policy_id!r} effective {mode} scale is {value}, "
                        f"expected {expected}"
                    )
        report[policy_id] = {
            "requested": True,
            "requested_scale": expected,
            "before_learner": before["learner"],
            "before_actor": before["actor"],
            "effective_learner": after["learner"],
            "effective_actor": after["actor"],
        }
    return report


class _MatrixSimulatorEvalRunner(SimulatorEvalRunner):
    def __init__(
        self,
        *args: Any,
        scoring_mode: str,
        greedy_policy_ids: Sequence[str] = (),
        action_rng_salt_mode: str = "shared",
        trace_path: Path | None = None,
        trace_top_k: int = 5,
        trace_max_decisions_per_episode: int = 0,
        trace_tensors: bool = False,
        action_catalog: ActionCatalog | None = None,
        force_pass_seat: int | None = None,
        force_pass_max_per_episode: int = 0,
        force_action_seat: int | None = None,
        force_action_decision_index: int | None = None,
        force_action_id: int | None = None,
        force_action_pair_index: int | None = None,
        force_action_swap_index: int | None = None,
        force_action_sequence: Sequence[Mapping[str, int]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._matrix_scoring_mode = str(scoring_mode)
        self._greedy_policy_ids = frozenset(str(policy_id) for policy_id in greedy_policy_ids)
        self._action_rng_salt_mode = str(action_rng_salt_mode)
        self._trace_path = trace_path
        self._trace_top_k = max(1, int(trace_top_k))
        self._trace_max_decisions_per_episode = max(0, int(trace_max_decisions_per_episode))
        self._trace_tensors = bool(trace_tensors)
        self._action_catalog = action_catalog
        self._force_pass_seat = None if force_pass_seat is None else int(force_pass_seat)
        self._force_pass_max_per_episode = max(0, int(force_pass_max_per_episode))
        self._force_action_seat = None if force_action_seat is None else int(force_action_seat)
        self._force_action_decision_index = (
            None if force_action_decision_index is None else int(force_action_decision_index)
        )
        self._force_action_id = None if force_action_id is None else int(force_action_id)
        self._force_action_pair_index = None if force_action_pair_index is None else int(force_action_pair_index)
        self._force_action_swap_index = None if force_action_swap_index is None else int(force_action_swap_index)
        sequence_entries: dict[tuple[int | None, int, int], int] = {}
        for item in force_action_sequence:
            swap_value = item.get("swap_index")
            key = (
                None if swap_value is None else int(swap_value),
                int(item["seat"]),
                int(item["decision_index"]),
            )
            sequence_entries[key] = int(item["action_id"])
        self._force_action_sequence = sequence_entries
        self._active_forced_pass_count = 0
        self._active_scheduled_game: ScheduledGame | None = None
        self._active_decision_index = 0
        self._matrix_counters: dict[str, int] = {
            "model_decisions": 0,
            "heuristic_decisions": 0,
            "random_legal_decisions": 0,
            "sample_decisions": 0,
            "greedy_override_decisions": 0,
            "fallback_to_parent_decisions": 0,
            "trace_rows": 0,
            "forced_pass_decisions": 0,
            "forced_action_decisions": 0,
            "forced_action_missed_decisions": 0,
            "force_action_sequence_entries": len(self._force_action_sequence),
        }

    def matrix_counters(self) -> dict[str, Any]:
        return {
            **self._matrix_counters,
            "scoring_mode": self._matrix_scoring_mode,
            "greedy_policy_ids": sorted(self._greedy_policy_ids),
            "greedy_override_requested": bool(self._greedy_policy_ids),
            "action_rng_salt_mode": self._action_rng_salt_mode,
            "trace_path": None if self._trace_path is None else self._trace_path.as_posix(),
            "trace_top_k": self._trace_top_k,
            "trace_max_decisions_per_episode": self._trace_max_decisions_per_episode,
            "trace_tensors": self._trace_tensors,
            "force_pass_seat": self._force_pass_seat,
            "force_pass_max_per_episode": self._force_pass_max_per_episode,
            "force_action_seat": self._force_action_seat,
            "force_action_decision_index": self._force_action_decision_index,
            "force_action_id": self._force_action_id,
            "force_action_pair_index": self._force_action_pair_index,
            "force_action_swap_index": self._force_action_swap_index,
            "force_action_sequence_entries": [
                {
                    "swap_index": key[0],
                    "seat": key[1],
                    "decision_index": key[2],
                    "action_id": action_id,
                }
                for key, action_id in sorted(
                    self._force_action_sequence.items(),
                    key=lambda item: (-1 if item[0][0] is None else int(item[0][0]), item[0][1], item[0][2]),
                )
            ],
        }

    def run_game(self, scheduled_game: ScheduledGame):  # type: ignore[override]
        self._active_scheduled_game = scheduled_game
        self._active_decision_index = 0
        self._active_forced_pass_count = 0
        try:
            return super().run_game(scheduled_game)
        finally:
            self._active_scheduled_game = None
            self._active_decision_index = 0
            self._active_forced_pass_count = 0

    def _forward_model_logits(
        self,
        *,
        policy: ResolvedEvalPolicy,
        batch: Any,
        current_seat: int,
        seat_hidden: torch.Tensor,
        scoring_mode: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        if policy.model is None:
            raise RuntimeError("model policy required")
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([int(current_seat)], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode=scoring_mode,
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return logits, next_seat_hidden

    def _forward_raw_and_final_logits(
        self,
        *,
        policy: ResolvedEvalPolicy,
        batch: Any,
        current_seat: int,
        seat_hidden: torch.Tensor,
    ) -> tuple[np.ndarray | None, np.ndarray, torch.Tensor, dict[str, Any]]:
        if policy.model is None:
            raise RuntimeError("model policy required")
        get_bias_scale = getattr(policy.model, "get_public_heuristic_logit_bias_scale", None)
        set_bias_scale = getattr(policy.model, "set_public_heuristic_logit_bias_scale", None)
        bias_report: dict[str, Any] = {}
        raw_logits: np.ndarray | None = None
        if callable(get_bias_scale) and callable(set_bias_scale):
            learner_scale = float(get_bias_scale(scoring_mode="learner"))
            actor_scale = float(get_bias_scale(scoring_mode="actor"))
            bias_report = {
                "effective_learner": learner_scale,
                "effective_actor": actor_scale,
            }
            try:
                set_bias_scale(0.0, actor_value=0.0)
                raw_logits, _raw_next_hidden = self._forward_model_logits(
                    policy=policy,
                    batch=batch,
                    current_seat=current_seat,
                    seat_hidden=seat_hidden,
                    scoring_mode=self._matrix_scoring_mode,
                )
            finally:
                set_bias_scale(learner_scale, actor_value=actor_scale)
        final_logits, next_hidden = self._forward_model_logits(
            policy=policy,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            scoring_mode=self._matrix_scoring_mode,
        )
        return raw_logits, final_logits, next_hidden, bias_report

    def _maybe_trace_model_decision(
        self,
        *,
        current_policy_id: str,
        batch: Any,
        current_seat: int,
        legal_ids: np.ndarray,
        selected_action: int,
        raw_logits: np.ndarray | None,
        final_logits: np.ndarray,
        bias_report: Mapping[str, Any],
        mode: str,
    ) -> None:
        if self._trace_path is None:
            return
        scheduled_game = self._active_scheduled_game
        if scheduled_game is None:
            return
        decision_index = int(self._active_decision_index)
        if self._trace_max_decisions_per_episode and decision_index >= self._trace_max_decisions_per_episode:
            return
        legal = np.asarray(legal_ids, dtype=np.uint32)
        selected_payload = _action_payload(int(selected_action), action_catalog=self._action_catalog)
        final_topk = _topk_payload(
            logits=final_logits,
            legal_ids=legal,
            top_k=self._trace_top_k,
            action_catalog=self._action_catalog,
        )
        raw_topk = (
            []
            if raw_logits is None
            else _topk_payload(
                logits=raw_logits,
                legal_ids=legal,
                top_k=self._trace_top_k,
                action_catalog=self._action_catalog,
            )
        )
        final_top_action = final_topk[0] if final_topk else {}
        raw_top_action = raw_topk[0] if raw_topk else {}
        obs_row = np.asarray(batch.obs[0], dtype=np.float32)
        row = {
            "pair_index": int(scheduled_game.pair_index),
            "swap_index": int(scheduled_game.swap_index),
            "episode_index": int(scheduled_game.episode_index),
            "episode_seed": int(scheduled_game.episode_seed),
            "decision_index": decision_index,
            "decision_id": int(np.asarray(batch.decision_id, dtype=np.int64)[0]),
            "actor_seat": int(current_seat),
            "policy_id": current_policy_id,
            "focal_policy_id": scheduled_game.focal_policy_id,
            "opponent_policy_id": scheduled_game.opponent_policy_id,
            "seat0_policy_id": scheduled_game.seat0_policy_id,
            "seat1_policy_id": scheduled_game.seat1_policy_id,
            "scoring_mode": self._matrix_scoring_mode,
            "selection_mode": mode,
            "legal_action_count": int(legal.size),
            "legal_ids": [int(item) for item in legal.tolist()],
            "legal_ids_sha256": hashlib.sha256(legal.tobytes()).hexdigest(),
            "selected_action": selected_payload,
            "raw_topk_no_public_bias": raw_topk,
            "final_topk": final_topk,
            "raw_top_action_matches_final": bool(raw_top_action.get("action_id") == final_top_action.get("action_id"))
            if raw_top_action and final_top_action
            else None,
            "raw_top_family_matches_final": bool(raw_top_action.get("family") == final_top_action.get("family"))
            if raw_top_action and final_top_action
            else None,
            "public_bias_report": dict(bias_report),
        }
        if self._trace_tensors:
            legal_i64 = legal.astype(np.int64, copy=False)
            row.update(
                {
                    "obs_sha256": hashlib.sha256(obs_row.tobytes()).hexdigest(),
                    "obs_float32": [float(item) for item in obs_row.tolist()],
                    "final_legal_logits": [float(item) for item in final_logits[legal_i64].tolist()],
                    "raw_legal_logits_no_public_bias": []
                    if raw_logits is None
                    else [float(item) for item in raw_logits[legal_i64].tolist()],
                }
            )
        _append_jsonl(self._trace_path, row)
        self._matrix_counters["trace_rows"] += 1

    def _maybe_force_pass(
        self,
        *,
        current_seat: int,
        legal_ids: np.ndarray,
        selected_action: int,
        mode: str,
    ) -> tuple[int, str]:
        if self._force_pass_seat is None or int(current_seat) != int(self._force_pass_seat):
            return int(selected_action), mode
        if self._force_pass_max_per_episode and self._active_forced_pass_count >= self._force_pass_max_per_episode:
            return int(selected_action), mode
        legal = np.asarray(legal_ids, dtype=np.int64)
        if int(self.pass_action_id) not in set(int(item) for item in legal.tolist()):
            return int(selected_action), mode
        if not np.any(legal != int(self.pass_action_id)):
            return int(selected_action), mode
        if int(selected_action) == int(self.pass_action_id):
            return int(selected_action), mode
        self._active_forced_pass_count += 1
        self._matrix_counters["forced_pass_decisions"] += 1
        return int(self.pass_action_id), f"{mode}+forced_pass"

    def _maybe_force_specific_action(
        self,
        *,
        current_seat: int,
        legal_ids: np.ndarray,
        selected_action: int,
        mode: str,
    ) -> tuple[int, str]:
        scheduled_game = self._active_scheduled_game
        if scheduled_game is None:
            return int(selected_action), mode
        sequence_key = (int(scheduled_game.swap_index), int(current_seat), int(self._active_decision_index))
        wildcard_key = (None, int(current_seat), int(self._active_decision_index))
        forced_action_id: int | None = None
        from_sequence = False
        if sequence_key in self._force_action_sequence:
            forced_action_id = int(self._force_action_sequence[sequence_key])
            from_sequence = True
        elif wildcard_key in self._force_action_sequence:
            forced_action_id = int(self._force_action_sequence[wildcard_key])
            from_sequence = True
        if forced_action_id is None:
            if (
                self._force_action_seat is None
                or self._force_action_decision_index is None
                or self._force_action_id is None
            ):
                return int(selected_action), mode
            forced_action_id = int(self._force_action_id)
            if (
                self._force_action_pair_index is not None
                and int(scheduled_game.pair_index) != self._force_action_pair_index
            ):
                return int(selected_action), mode
            if (
                self._force_action_swap_index is not None
                and int(scheduled_game.swap_index) != self._force_action_swap_index
            ):
                return int(selected_action), mode
            if int(current_seat) != self._force_action_seat:
                return int(selected_action), mode
            if int(self._active_decision_index) != self._force_action_decision_index:
                return int(selected_action), mode
        elif not from_sequence:
            return int(selected_action), mode
        legal = {int(item) for item in np.asarray(legal_ids, dtype=np.int64).tolist()}
        if int(forced_action_id) not in legal:
            self._matrix_counters["forced_action_missed_decisions"] += 1
            return int(selected_action), f"{mode}+force_action_missed"
        self._matrix_counters["forced_action_decisions"] += 1
        return int(forced_action_id), f"{mode}+forced_action"

    def _select_action(self, **kwargs: Any) -> tuple[int, torch.Tensor | None]:  # type: ignore[override]
        current_policy_id = str(kwargs.get("current_policy_id"))
        policy = self.policies.get(current_policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {current_policy_id!r}")
        if policy.heuristic_policy is not None:
            self._matrix_counters["heuristic_decisions"] += 1
            action = policy.heuristic_policy.choose_action(
                np.asarray(kwargs["batch"].obs[0], dtype=np.float32),
                np.asarray(kwargs["legal_ids"], dtype=np.uint32),
            )
            self._active_decision_index += 1
            return int(action), kwargs.get("seat_hidden")
        if policy.model is None:
            self._matrix_counters["random_legal_decisions"] += 1
            self._matrix_counters["sample_decisions"] += 1
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                np.asarray(kwargs["legal_ids"], dtype=np.uint32),
                rng=kwargs["rng"],
            )
            self._active_decision_index += 1
            return int(action), kwargs.get("seat_hidden")
        seat_hidden = kwargs.get("seat_hidden")
        if seat_hidden is None:
            self._matrix_counters["fallback_to_parent_decisions"] += 1
            return super()._select_action(**kwargs)
        batch = kwargs["batch"]
        current_seat = int(kwargs["current_seat"])
        legal_ids = np.asarray(kwargs["legal_ids"], dtype=np.uint32)
        if legal_ids.size == 0:
            self._matrix_counters["fallback_to_parent_decisions"] += 1
            return super()._select_action(**kwargs)
        self._matrix_counters["model_decisions"] += 1
        raw_logits, logits, next_seat_hidden, bias_report = self._forward_raw_and_final_logits(
            policy=policy,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
        )
        if current_policy_id in self._greedy_policy_ids:
            self._matrix_counters["greedy_override_decisions"] += 1
            legal_logits = logits[legal_ids.astype(np.int64, copy=False)]
            selected_action = int(legal_ids[int(np.argmax(legal_logits))])
            selected_action, trace_mode = self._maybe_force_pass(
                current_seat=current_seat,
                legal_ids=legal_ids,
                selected_action=selected_action,
                mode="greedy",
            )
            selected_action, trace_mode = self._maybe_force_specific_action(
                current_seat=current_seat,
                legal_ids=legal_ids,
                selected_action=selected_action,
                mode=trace_mode,
            )
            self._maybe_trace_model_decision(
                current_policy_id=current_policy_id,
                batch=batch,
                current_seat=current_seat,
                legal_ids=legal_ids,
                selected_action=selected_action,
                raw_logits=raw_logits,
                final_logits=logits,
                bias_report=bias_report,
                mode=trace_mode,
            )
            self._active_decision_index += 1
            return selected_action, next_seat_hidden
        self._matrix_counters["sample_decisions"] += 1
        action, _logp = sample_action_pinned(
            logits,
            legal_ids,
            rng=kwargs["rng"],
        )
        selected_action = int(action)
        selected_action, trace_mode = self._maybe_force_pass(
            current_seat=current_seat,
            legal_ids=legal_ids,
            selected_action=selected_action,
            mode="sample",
        )
        selected_action, trace_mode = self._maybe_force_specific_action(
            current_seat=current_seat,
            legal_ids=legal_ids,
            selected_action=selected_action,
            mode=trace_mode,
        )
        self._maybe_trace_model_decision(
            current_policy_id=current_policy_id,
            batch=batch,
            current_seat=current_seat,
            legal_ids=legal_ids,
            selected_action=selected_action,
            raw_logits=raw_logits,
            final_logits=logits,
            bias_report=bias_report,
            mode=trace_mode,
        )
        self._active_decision_index += 1
        return selected_action, next_seat_hidden

    def _rng_seed(self, *, scheduled_game: ScheduledGame, seat: int) -> int:
        base = super()._rng_seed(scheduled_game=scheduled_game, seat=seat)
        mode = self._action_rng_salt_mode
        if mode == "shared":
            return int(base)
        if mode == "physical":
            return stable_hash64(
                canonical_json_bytes(
                    {
                        "kind": "b1_artifact_matrix_physical_action_rng_v1",
                        "pair_index": int(scheduled_game.pair_index),
                        "swap_index": int(scheduled_game.swap_index),
                        "episode_seed": int(scheduled_game.episode_seed),
                        "seat": int(seat),
                    }
                )
            )
        salt_payload: dict[str, Any] = {
            "kind": "b1_artifact_matrix_action_rng_salt_v1",
            "base": int(base),
            "mode": mode,
            "seat": int(seat),
            "pair_index": int(scheduled_game.pair_index),
            "swap_index": int(scheduled_game.swap_index),
            "episode_seed": int(scheduled_game.episode_seed),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
        if mode == "matchup":
            salt_payload["focal_policy_id"] = scheduled_game.focal_policy_id
            salt_payload["opponent_policy_id"] = scheduled_game.opponent_policy_id
        elif mode != "policy":
            raise RuntimeError(f"unsupported action_rng_salt_mode {mode!r}")
        return stable_hash64(canonical_json_bytes(salt_payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", action="append", type=_parse_policy_arg, default=[])
    parser.add_argument("--include-builtin", action="append", type=_canonical_builtin_policy_id, default=[])
    parser.add_argument("--matchup", action="append", type=_parse_matchup_arg, default=[])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--artifact-dir-name", default="b1_artifact_matrix")
    parser.add_argument(
        "--surface-name", default="", help="Optional eval-surface label, e.g. official_s3/lowbias_s1/raw_s0."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--focal-action-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--both-greedy", action="store_true")
    parser.add_argument("--disable-public-heuristic-bias", action="store_true")
    parser.add_argument("--public-heuristic-bias-scale", type=float, default=None)
    parser.add_argument("--scoring-mode", choices=("learner", "actor"), default="learner")
    parser.add_argument("--seed-scope", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--action-rng-salt-mode", choices=("shared", "physical", "policy", "matchup"), default="shared")
    parser.add_argument("--emit-action-traces", action="store_true")
    parser.add_argument("--emit-trace-tensors", action="store_true")
    parser.add_argument("--trace-top-k", type=int, default=5)
    parser.add_argument("--trace-max-decisions-per-episode", type=int, default=0)
    parser.add_argument("--force-pass-seat", type=int, choices=(0, 1), default=None)
    parser.add_argument("--force-pass-max-per-episode", type=int, default=0)
    parser.add_argument("--force-action-seat", type=int, choices=(0, 1), default=None)
    parser.add_argument("--force-action-decision-index", type=int, default=None)
    parser.add_argument("--force-action-id", type=int, default=None)
    parser.add_argument("--force-action-pair-index", type=int, default=None)
    parser.add_argument("--force-action-swap-index", type=int, choices=(0, 1), default=None)
    parser.add_argument("--force-action-sequence", action="append", type=_parse_force_action_sequence, default=[])
    args = parser.parse_args()

    if not args.checkpoint_policy:
        raise SystemExit("provide at least one --checkpoint-policy alias=checkpoint.pt")
    if int(args.pairs) <= 0:
        raise SystemExit("--pairs must be positive")
    if args.disable_public_heuristic_bias and args.public_heuristic_bias_scale is not None:
        raise SystemExit("use either --disable-public-heuristic-bias or --public-heuristic-bias-scale, not both")
    bias_override_scale = 0.0 if args.disable_public_heuristic_bias else args.public_heuristic_bias_scale

    stack = train_script.load_stack_config(args.stack_config)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_spec = manifest.get("runtime_spec")
    spec_hash = str(runtime_spec.get("sha256") or "") if isinstance(runtime_spec, dict) else ""
    spec_hash = spec_hash or _manifest_value(manifest, "spec_hash256", "spec_hash")
    contract = train_script.load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    seed_file, validated_sources, base_seeds, seed_sha = train_script._periodic_dev_eval_schedule(stack)
    if int(args.seed_offset) != 0 and base_seeds:
        offset = int(args.seed_offset) % len(base_seeds)
        base_seeds = tuple(base_seeds[offset:]) + tuple(base_seeds[:offset])
    paired_seeds = train_script._expand_periodic_dev_eval_paired_seeds(
        base_seeds,
        requested_pairs=int(args.pairs),
        seed_file_sha256=seed_sha,
        update_count=int(args.seed_offset),
        policy_version=int(args.seed_offset),
        scope=str(args.seed_scope or args.artifact_dir_name),
    )
    train_script._validate_periodic_dev_eval_contract(stack)
    evaluation = train_script._evaluation_config_or_raise(stack)
    observation_dim, action_dim = train_script._spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    observation_spec = contract.spec_bundle.get("observation")
    spec_bundle = contract.spec_bundle
    action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
    artifact_layout = ArtifactLayout.from_run_dir(args.run_dir)

    policies: dict[str, ResolvedEvalPolicy] = {}
    source_paths: dict[str, Path | None] = {}
    b1_resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID],
        run_dir=args.run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=spec_bundle,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        eval_device=args.device,
    )[NO_LEAGUE_POLICY_ID]
    policies[NO_LEAGUE_POLICY_ID] = b1_resolved
    source_paths[NO_LEAGUE_POLICY_ID] = (
        None
        if b1_resolved.source_run_dir is None or b1_resolved.snapshot_path is None
        else Path(b1_resolved.source_run_dir) / b1_resolved.snapshot_path
    )
    builtin_policy_ids = []
    for policy_id in args.include_builtin:
        if policy_id == NO_LEAGUE_POLICY_ID:
            continue
        if policy_id not in builtin_policy_ids:
            builtin_policy_ids.append(policy_id)
    if builtin_policy_ids:
        resolved_builtins = resolve_eval_policies(
            stack=stack,
            policy_ids=builtin_policy_ids,
            run_dir=args.run_dir,
            observation_dim=observation_dim,
            action_dim=action_dim,
            spec_bundle=spec_bundle,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            eval_device=args.device,
        )
        for policy_id in builtin_policy_ids:
            policies[policy_id] = resolved_builtins[policy_id]
            source_paths[policy_id] = None

    for alias, checkpoint_path in args.checkpoint_policy:
        checkpoint_path = checkpoint_path.resolve()
        model = train_script._load_checkpoint_eval_model(
            checkpoint_path=checkpoint_path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=args.device,
            observation_spec=observation_spec if isinstance(observation_spec, dict) else None,
            spec_bundle=spec_bundle if isinstance(spec_bundle, dict) else None,
        )
        policies[alias] = ResolvedEvalPolicy(
            policy_id=alias,
            kind="checkpoint",
            source_run_dir=checkpoint_path.parent.parent.parent.as_posix(),
            snapshot_path=checkpoint_path.as_posix(),
            model=model,
        )
        source_paths[alias] = checkpoint_path

    bias_override_report = _apply_public_heuristic_bias_override(
        policies,
        override_scale=None if bias_override_scale is None else float(bias_override_scale),
    )

    policy_ids = tuple(policies)
    matrix_dir = args.run_dir / "eval" / args.artifact_dir_name
    matrix_dir.mkdir(parents=True, exist_ok=True)

    load_manifest = {
        "format": "b1_artifact_matrix_load_manifest_v1",
        "run_dir": args.run_dir.as_posix(),
        "stack_config": str(args.stack_config),
        "surface_name": str(args.surface_name or ""),
        "pairs": int(args.pairs),
        "focal_action_mode": str(args.focal_action_mode),
        "both_greedy": bool(args.both_greedy),
        "scoring_mode": str(args.scoring_mode),
        "action_rng_salt_mode": str(args.action_rng_salt_mode),
        "emit_action_traces": bool(args.emit_action_traces),
        "emit_trace_tensors": bool(args.emit_trace_tensors),
        "trace_top_k": int(args.trace_top_k),
        "trace_max_decisions_per_episode": int(args.trace_max_decisions_per_episode),
        "force_pass_seat": args.force_pass_seat,
        "force_pass_max_per_episode": int(args.force_pass_max_per_episode),
        "force_action_seat": args.force_action_seat,
        "force_action_decision_index": args.force_action_decision_index,
        "force_action_id": args.force_action_id,
        "force_action_pair_index": args.force_action_pair_index,
        "force_action_swap_index": args.force_action_swap_index,
        "force_action_sequence": [item for sequence in args.force_action_sequence for item in sequence],
        "seed_scope": str(args.seed_scope or args.artifact_dir_name),
        "seed_offset": int(args.seed_offset),
        "public_heuristic_bias_override_requested": bias_override_scale is not None,
        "public_heuristic_bias_override_scale": bias_override_scale,
        "paired_seeds": list(paired_seeds),
        "seed_file": {
            "path": seed_file.as_posix(),
            "sha256": seed_sha,
            "validated_sources": dict(validated_sources),
        },
        "policies": {
            policy_id: _policy_manifest_entry(
                policy_id=policy_id,
                policy=policy,
                source_path=source_paths.get(policy_id),
            )
            for policy_id, policy in policies.items()
        },
        "public_heuristic_bias_override_report": bias_override_report,
        "pairwise_model_distances": {},
    }
    for left in policy_ids:
        for right in policy_ids:
            if left >= right:
                continue
            left_model = policies[left].model
            right_model = policies[right].model
            if left_model is None or right_model is None:
                continue
            load_manifest["pairwise_model_distances"][f"{left}__vs__{right}"] = _state_dict_l2_distance(
                left_model,
                right_model,
            )
    _write_json(matrix_dir / "policy_load_manifest.json", load_manifest)
    _write_json(
        matrix_dir / "resolved_policies.json", {key: policy.to_manifest_dict() for key, policy in policies.items()}
    )

    matchup_matrix: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_pairs: list[tuple[str, str]] = []
    if args.matchup:
        for raw_focal, raw_opponent in args.matchup:
            ordered_pairs.append(
                (
                    _resolve_policy_alias(raw_focal, policy_ids),
                    _resolve_policy_alias(raw_opponent, policy_ids),
                )
            )
    else:
        for focal_policy_id in policy_ids:
            for opponent_policy_id in policy_ids:
                if focal_policy_id == opponent_policy_id and not args.include_self:
                    continue
                ordered_pairs.append((focal_policy_id, opponent_policy_id))

    for focal_policy_id, opponent_policy_id in ordered_pairs:
        matchup_dir = matrix_dir / f"{_safe_slug(focal_policy_id)}__vs__{_safe_slug(opponent_policy_id)}"
        trace_path = matchup_dir / "action_trace.jsonl" if args.emit_action_traces else None
        if trace_path is not None and trace_path.exists():
            trace_path.unlink()
        greedy_policy_ids: list[str] = []
        if args.focal_action_mode == "greedy":
            greedy_policy_ids.append(focal_policy_id)
        if args.both_greedy:
            greedy_policy_ids.extend([focal_policy_id, opponent_policy_id])
        greedy_policy_ids = list(dict.fromkeys(greedy_policy_ids))
        runner = _MatrixSimulatorEvalRunner(
            stack=stack,
            policies=policies,
            artifact_layout=artifact_layout,
            run_id256=_manifest_value(manifest, "run_id256", "computed_run_id256"),
            spec_hash256=spec_hash,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
            replay_capture_rate=0.0,
            regression_capture_count=0,
            eval_device=args.device,
            spec_bundle=spec_bundle if isinstance(spec_bundle, dict) else None,
            scoring_mode=str(args.scoring_mode),
            greedy_policy_ids=greedy_policy_ids,
            action_rng_salt_mode=str(args.action_rng_salt_mode),
            trace_path=trace_path,
            trace_top_k=int(args.trace_top_k),
            trace_max_decisions_per_episode=int(args.trace_max_decisions_per_episode),
            trace_tensors=bool(args.emit_trace_tensors),
            action_catalog=action_catalog,
            force_pass_seat=args.force_pass_seat,
            force_pass_max_per_episode=int(args.force_pass_max_per_episode),
            force_action_seat=args.force_action_seat,
            force_action_decision_index=args.force_action_decision_index,
            force_action_id=args.force_action_id,
            force_action_pair_index=args.force_action_pair_index,
            force_action_swap_index=args.force_action_swap_index,
            force_action_sequence=[item for sequence in args.force_action_sequence for item in sequence],
        )
        runner_counters: dict[str, Any] = {}
        try:
            result = run_seat_swapped_matchup(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                paired_seeds=paired_seeds,
                runner=runner,
                episodes_path=matchup_dir / "episodes.jsonl",
                run_id256=_manifest_value(manifest, "run_id256", "computed_run_id256"),
                config_hash256=_manifest_value(manifest, "config_hash256", "config_hash"),
                spec_hash256=spec_hash,
            )
            runner_counters = runner.matrix_counters()
        finally:
            runner.close()
        greedy_model_requested = any(policies[policy_id].model is not None for policy_id in greedy_policy_ids)
        if greedy_model_requested and int(runner_counters.get("greedy_override_decisions", 0)) <= 0:
            raise RuntimeError(
                f"greedy override was requested for {greedy_policy_ids}, but no model decisions were overridden "
                f"in matchup {focal_policy_id!r} vs {opponent_policy_id!r}"
            )
        matchup_payload = build_matchup_export(
            list(result.records),
            stop_rules=evaluation.stop_rules,
            max_paired_seeds=len(paired_seeds),
            scheme=evaluation.final_policy_set_selection.folding,
            sample_count=1000,
            seed=0,
        )
        seat_diagnostics = build_seat_advantage_diagnostics(list(result.records))
        pair_rows, pair_summary = _pair_table(
            list(result.records),
            scheme=evaluation.final_policy_set_selection.folding,
        )
        matchup_payload["seat_diagnostics"] = seat_diagnostics
        matchup_payload["pair_class_summary"] = pair_summary
        matchup_payload["matrix_runner_counters"] = runner_counters
        matchup_payload["evaluation_context"] = {
            "artifact_dir_name": str(args.artifact_dir_name),
            "surface_name": str(args.surface_name or ""),
            "focal_policy_id": focal_policy_id,
            "opponent_policy_id": opponent_policy_id,
            "focal_action_mode": str(args.focal_action_mode),
            "both_greedy": bool(args.both_greedy),
            "scoring_mode": str(args.scoring_mode),
            "action_rng_salt_mode": str(args.action_rng_salt_mode),
            "public_heuristic_bias_override_requested": bias_override_scale is not None,
            "public_heuristic_bias_override_scale": bias_override_scale,
            "force_pass_seat": args.force_pass_seat,
            "force_pass_max_per_episode": int(args.force_pass_max_per_episode),
            "force_action_seat": args.force_action_seat,
            "force_action_decision_index": args.force_action_decision_index,
            "force_action_id": args.force_action_id,
            "force_action_pair_index": args.force_action_pair_index,
            "force_action_swap_index": args.force_action_swap_index,
            "force_action_sequence": [item for sequence in args.force_action_sequence for item in sequence],
            "episodes_path": (matchup_dir / "episodes.jsonl").relative_to(args.run_dir).as_posix(),
            "pair_table_path": (matchup_dir / "pair_table.jsonl").relative_to(args.run_dir).as_posix(),
            "action_trace_path": None if trace_path is None else trace_path.relative_to(args.run_dir).as_posix(),
        }
        write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
        _write_json(matchup_dir / "seat_diagnostics.json", seat_diagnostics)
        _write_json(matchup_dir / "pair_class_summary.json", pair_summary)
        _write_jsonl(matchup_dir / "pair_table.jsonl", pair_rows)
        matchup_matrix[(focal_policy_id, opponent_policy_id)] = {
            "focal_policy_id": focal_policy_id,
            "opponent_policy_id": opponent_policy_id,
            "matchup_dir": matchup_dir.relative_to(args.run_dir).as_posix(),
            "matchup_payload": matchup_payload,
        }
        print(
            f"{focal_policy_id} vs {opponent_policy_id}: "
            f"mean={matchup_payload['uncertainty']['mean']} "
            f"wins={matchup_payload['summary']['wins']} losses={matchup_payload['summary']['losses']} "
            f"pair_classes={pair_summary['pair_class_counts']}"
        )

    complement_checks = []
    for left in policy_ids:
        for right in policy_ids:
            if left >= right:
                continue
            check = _complement_check(matchup_matrix, left=left, right=right)
            if check is not None:
                complement_checks.append(check)
    matrix_summary = {
        "format": "b1_artifact_matrix_summary_v1",
        "surface_name": str(args.surface_name or ""),
        "scoring_mode": str(args.scoring_mode),
        "public_heuristic_bias_override_requested": bias_override_scale is not None,
        "public_heuristic_bias_override_scale": bias_override_scale,
        "policy_ids": list(policy_ids),
        "pairs": int(args.pairs),
        "matchups": {
            f"{left}__vs__{right}": {
                "matchup_dir": payload["matchup_dir"],
                "mean": payload["matchup_payload"]["uncertainty"]["mean"],
                "wins": payload["matchup_payload"]["summary"]["wins"],
                "losses": payload["matchup_payload"]["summary"]["losses"],
                "draws": payload["matchup_payload"]["summary"]["draws"],
                "truncations": payload["matchup_payload"]["summary"]["truncations"],
                "pair_class_summary": payload["matchup_payload"]["pair_class_summary"],
                "seat_diagnostics": payload["matchup_payload"]["seat_diagnostics"],
                "matrix_runner_counters": payload["matchup_payload"]["matrix_runner_counters"],
            }
            for (left, right), payload in matchup_matrix.items()
        },
        "complement_checks": complement_checks,
    }
    _write_json(matrix_dir / "matrix_summary.json", matrix_summary)
    print(f"wrote matrix artifacts to {matrix_dir}")


if __name__ == "__main__":
    main()
