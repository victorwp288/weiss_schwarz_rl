"""Shared human-vs-model play session engine."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.diagnostics.action_diagnostics import (
    ActionSummaryCounters,
    make_action_sequence_state,
    summarize_eval_action_counters,
    update_eval_action_counters,
)
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval import load_dev_eval_summaries
from weiss_rl.eval.god_search import GodSearchConfig
from weiss_rl.eval.harness import ScheduledGame, game_result_from_step
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    LEGACY_NO_LEAGUE_POLICY_ID,
    MAIN_DECK_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
    deck_id_for_policy_id,
    recommend_focal_policy_id,
)
from weiss_rl.eval.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.human_play.transcript import DecisionRecord, HumanPlayTranscript
from weiss_rl.league.registry import SnapshotRegistry


class HumanPlaySessionError(RuntimeError):
    """Raised when a human-play session cannot continue safely."""


@dataclass(frozen=True, slots=True)
class HumanPlayConfig:
    run_dir: Path
    policy_id: str = "main_league_selected"
    stack_config: Path | None = None
    snapshot_registry_json: Path | None = None
    b1_baseline_run_dir: Path | None = None
    human_seat: int = 0
    seed: int = 20260521
    human_deck: str = MAIN_DECK_ID
    model_deck: str = MAIN_DECK_ID
    mode: str = "study"
    model_sampling_algorithm: str = "model_argmax_pinned_v1"
    artifact_root: Path | None = None
    top_k: int = 5
    search_rollout_opponent_policy_id: str = RANDOM_LEGAL_POLICY_ID
    god_search: GodSearchConfig = GodSearchConfig()

    def __post_init__(self) -> None:
        if int(self.human_seat) not in {0, 1}:
            raise ValueError("human_seat must be 0 or 1")
        if str(self.mode) not in {"study", "freeplay"}:
            raise ValueError("mode must be 'study' or 'freeplay'")


@dataclass(frozen=True, slots=True)
class ActionOption:
    action_id: int
    label: str
    family: str | None = None
    probability: float | None = None
    logit: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "label": self.label,
            "family": self.family,
            "probability": self.probability,
            "logit": self.logit,
        }


class HumanPlaySession:
    """One live simulator session shared by web and future UI frontends."""

    def __init__(self, config: HumanPlayConfig) -> None:
        self.config = config
        self.session_id = uuid.uuid4().hex
        self.weiss_sim = _require_human_view_api()
        self.run_dir = Path(config.run_dir).resolve()
        self.stack = load_stack_config(_resolve_stack_config_path(self.run_dir, config.stack_config))
        self.contract = load_verified_simulator_contract(
            _repo_root_from_run_dir(self.run_dir),
            expected_spec_hash=_resolve_expected_spec_hash(self.run_dir),
        )
        self.spec_bundle = self.contract.spec_bundle
        self.observation_dim = int(self.spec_bundle["observation"]["obs_len"])
        self.action_dim = int(self.spec_bundle["action"]["action_space_size"])
        self.pass_action_id = int(self.spec_bundle["action"]["pass_action_id"])
        self.policy_id = _normalize_policy_id(self.run_dir, config.policy_id)
        self.search_rollout_opponent_policy_id = str(config.search_rollout_opponent_policy_id).strip()
        self.model_seat = 1 - int(config.human_seat)
        self.seat0_policy_id = "human" if config.human_seat == 0 else self.policy_id
        self.seat1_policy_id = self.policy_id if config.human_seat == 0 else "human"
        self.seat0_deck = _deck_for_seat(config=config, seat=0)
        self.seat1_deck = _deck_for_seat(config=config, seat=1)
        self.env = self._build_env()
        self.batch = self.env.reset(seed=int(config.seed))
        self.rng = Pcg32XshRrV1(int(config.seed) ^ 0xC0DEC0DE)
        self.runner = self._build_eval_runner()
        self.model_hidden = self.runner._initial_hidden(  # noqa: SLF001 - shared eval helper until public API exists.
            self.policy_id,
            opponent_policy_id=self.search_rollout_opponent_policy_id,
        )
        self.action_sequence_state = make_action_sequence_state(1)
        self.action_counters = ActionSummaryCounters()
        self.action_history: list[int] = []
        self.game_search_state: dict[str, int] = {"searched": 0}
        self.recent_model_actions: list[dict[str, Any]] = []
        self.decision_count = 0
        self.last_acting_seat: int | None = None
        self.started_at = time.time()
        self.transcript = self._create_transcript()
        self.transcript.append_event({"event": "session_started", "session_id": self.session_id})
        self.advance_until_human_or_terminal()

    def close(self) -> None:
        self.env.close()

    def current_state(self) -> dict[str, Any]:
        view = self.current_view()
        terminal = self._is_terminal()
        payload = {
            "session_id": self.session_id,
            "mode": self.config.mode,
            "human_seat": int(self.config.human_seat),
            "model_seat": int(self.model_seat),
            "policy_id": self.policy_id,
            "human_turn": self._current_actor() == int(self.config.human_seat) and not terminal,
            "terminal": terminal,
            "view": view,
            "model": {
                "recent_actions": list(self.recent_model_actions),
                "god_search": self.runner.god_search_diagnostics(),
            },
            "artifacts": {
                "session_dir": str(self.transcript.session_dir),
                "manifest": str(self.transcript.manifest_path),
                "decisions": str(self.transcript.decisions_path),
                "postgame_report": str(self.transcript.postgame_path),
            },
        }
        if terminal:
            payload["result"] = self._terminal_summary()
        return payload

    def current_view(self) -> dict[str, Any]:
        raw_view = self.weiss_sim.human_decision_view(
            self.env.pool,
            env_index=0,
            perspective_seat=int(self.config.human_seat),
        )
        if not isinstance(raw_view, dict):
            raise HumanPlaySessionError("weiss_sim.human_decision_view() must return a dict")
        return _enrich_card_labels(_normalize_view(raw_view), self.weiss_sim)

    def submit_human_action(self, action_id: int, *, client_view_hash64: str | None = None) -> dict[str, Any]:
        if self._is_terminal():
            raise HumanPlaySessionError("cannot submit an action after the game is terminal")
        current_actor = self._current_actor()
        if current_actor != int(self.config.human_seat):
            raise HumanPlaySessionError(f"it is not the human player's turn; actor seat is {current_actor}")
        before = self.current_view()
        if client_view_hash64 is not None and before.get("view_hash64") != str(client_view_hash64):
            raise HumanPlaySessionError("client view is stale; refresh before submitting an action")
        legal_ids = tuple(int(item) for item in before["legal_action_ids"])
        selected = int(action_id)
        if selected not in legal_ids:
            raise HumanPlaySessionError(f"illegal action_id {selected}; legal actions are {list(legal_ids)}")
        self._step_and_record(
            action=selected,
            actor_kind="human",
            before_view=before,
            ranked_actions=(),
            elapsed_ms=None,
        )
        self.advance_until_human_or_terminal()
        if self._is_terminal():
            self.transcript.write_postgame_report(self._terminal_summary())
        return self.current_state()

    def advance_until_human_or_terminal(self) -> None:
        while not self._is_terminal() and self._current_actor() != int(self.config.human_seat):
            before = self.current_view()
            legal_ids = self._legal_ids_for_row()
            started = time.perf_counter()
            action, next_hidden, ranked = self._select_model_action(legal_ids=legal_ids)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.model_hidden = next_hidden
            self._step_and_record(
                action=action,
                actor_kind="model",
                before_view=before,
                ranked_actions=ranked,
                elapsed_ms=elapsed_ms,
            )

    def _select_model_action(
        self, *, legal_ids: np.ndarray
    ) -> tuple[int, torch.Tensor | None, tuple[ActionOption, ...]]:
        ranked = self._rank_model_actions(legal_ids=legal_ids)
        if self.config.god_search.enabled:
            action, next_hidden = self.runner._select_action(  # noqa: SLF001
                batch=self.batch,
                current_seat=int(self.model_seat),
                current_policy_id=self.policy_id,
                opponent_policy_id=self.search_rollout_opponent_policy_id,
                seat_hidden=self.model_hidden,
                rng=self.rng,
                legal_ids=legal_ids,
                action_sequence_state=self.action_sequence_state,
                scheduled_game=self._scheduled_game(),
                action_history=tuple(self.action_history),
                seat_hidden_by_seat={int(self.model_seat): self.model_hidden, int(self.config.human_seat): None},
                game_search_state=self.game_search_state,
            )
        else:
            action, next_hidden = self.runner._select_action_without_god_search(  # noqa: SLF001
                batch=self.batch,
                current_seat=int(self.model_seat),
                current_policy_id=self.policy_id,
                opponent_policy_id=self.search_rollout_opponent_policy_id,
                seat_hidden=self.model_hidden,
                rng=self.rng,
                legal_ids=legal_ids,
                action_sequence_state=self.action_sequence_state,
                sampling_algorithm=self.config.model_sampling_algorithm,
            )
        return int(action), next_hidden, ranked

    def _rank_model_actions(self, *, legal_ids: np.ndarray) -> tuple[ActionOption, ...]:
        policy = self.runner.policies[self.policy_id]
        if policy.model is None or self.model_hidden is None:
            return ()
        logits, _next_hidden, legal_ids_for_model = self.runner._model_logits_for_eval(  # noqa: SLF001
            policy=policy,
            current_policy_id=self.policy_id,
            opponent_policy_id=self.search_rollout_opponent_policy_id,
            batch=self.batch,
            current_seat=int(self.model_seat),
            seat_hidden=self.model_hidden,
            legal_ids=legal_ids,
            action_sequence_state=self.action_sequence_state,
        )
        legal_ids_array = np.asarray(legal_ids_for_model, dtype=np.int64)
        if legal_ids_array.size == 0:
            return ()
        legal_logits = np.asarray(logits[legal_ids_array], dtype=np.float64)
        shifted = legal_logits - float(np.max(legal_logits))
        probs = np.exp(shifted)
        probs = probs / float(np.sum(probs))
        order = np.argsort(probs)[::-1][: max(1, int(self.config.top_k))]
        view_by_action = {int(item.get("action_id")): item for item in self.current_view()["legal_actions"]}
        ranked = []
        for idx in order:
            action_id = int(legal_ids_array[int(idx)])
            view_action = view_by_action.get(action_id, {})
            ranked.append(
                ActionOption(
                    action_id=action_id,
                    label=str(view_action.get("label") or f"Action {action_id}"),
                    family=None if view_action.get("family") is None else str(view_action.get("family")),
                    probability=float(probs[int(idx)]),
                    logit=float(logits[action_id]),
                )
            )
        return tuple(ranked)

    def _step_and_record(
        self,
        *,
        action: int,
        actor_kind: str,
        before_view: dict[str, Any],
        ranked_actions: tuple[ActionOption, ...],
        elapsed_ms: float | None,
    ) -> None:
        legal_ids = tuple(int(item) for item in before_view["legal_action_ids"])
        actor_seat = int(before_view.get("summary", {}).get("actor_seat", self._current_actor()))
        self.last_acting_seat = actor_seat
        label = _label_for_action(before_view, action)
        ranked_payload = tuple(item.to_json_dict() for item in ranked_actions)
        self.transcript.append_decision(
            DecisionRecord(
                decision_index=self.decision_count,
                actor_seat=actor_seat,
                actor_kind=actor_kind,
                action_id=int(action),
                action_label=label,
                legal_action_ids=legal_ids,
                decision_id=_optional_int(before_view.get("decision_id")),
                decision_kind=_optional_str(before_view.get("summary", {}).get("decision_kind")),
                view_hash64=_optional_str(before_view.get("view_hash64")),
                legal_fingerprint64=_optional_str(before_view.get("legal_fingerprint64")),
                elapsed_ms=elapsed_ms,
                model_ranked_actions=ranked_payload,
            )
        )
        update_eval_action_counters(
            counters=self.action_counters,
            state=self.action_sequence_state,
            action=int(action),
            legal_ids=np.asarray(legal_ids, dtype=np.uint32),
            pass_action_id=self.pass_action_id,
        )
        if actor_kind == "model":
            self.recent_model_actions.append(
                {
                    "decision_index": int(self.decision_count),
                    "actor_seat": actor_seat,
                    "action_id": int(action),
                    "action_label": label,
                    "elapsed_ms": elapsed_ms,
                    "ranked_actions": list(ranked_payload),
                }
            )
            self.recent_model_actions = self.recent_model_actions[-8:]
        self.batch = self.env.step(np.asarray([int(action)], dtype=np.uint32))
        self.decision_count += 1
        self.action_history.append(int(action))

    def _build_env(self) -> DecisionBoundaryEnv:
        env_config = build_env_config_from_stack(
            self.stack,
            seed=int(self.config.seed),
            deck=self.seat0_deck,
            opponent_deck=self.seat1_deck,
        )
        pool, layout_name = make_env_pool_from_config(env_config, profile="fast", num_envs=1)
        if layout_name != "i16_legal_ids":
            raise HumanPlaySessionError(f"human play requires i16 legal ids, got {layout_name!r}")
        return DecisionBoundaryEnv(
            pool,
            legality="ids_offsets",
            pass_action_id=self.pass_action_id,
            engine_status_policy="hard_fail",
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
        )

    def _build_eval_runner(self) -> SimulatorEvalRunner:
        policy_ids = [self.policy_id]
        if self.config.god_search.enabled and self.search_rollout_opponent_policy_id not in policy_ids:
            policy_ids.append(self.search_rollout_opponent_policy_id)
        policies = resolve_eval_policies(
            stack=self.stack,
            policy_ids=policy_ids,
            run_dir=self.run_dir,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            spec_bundle=self.spec_bundle,
            snapshot_registry_path=self.config.snapshot_registry_json,
            b1_baseline_run_dir=self.config.b1_baseline_run_dir,
        )
        return SimulatorEvalRunner(
            stack=self.stack,
            policies=policies,
            artifact_layout=ArtifactLayout.from_run_dir(self.run_dir),
            run_id256=_resolve_run_id256(self.run_dir),
            spec_hash256=_resolve_expected_spec_hash(self.run_dir),
            action_dim=self.action_dim,
            pass_action_id=self.pass_action_id,
            require_sorted_legal_ids=True,
            replay_capture_rate=0.0,
            regression_capture_count=0,
            god_search_config=self.config.god_search,
        )

    def _create_transcript(self) -> HumanPlayTranscript:
        root = self.config.artifact_root or (self.run_dir / "human_play")
        session_dir = root / self.session_id
        return HumanPlayTranscript(
            session_dir,
            manifest={
                "session_id": self.session_id,
                "mode": self.config.mode,
                "run_dir": self.run_dir.as_posix(),
                "policy_id": self.policy_id,
                "human_seat": int(self.config.human_seat),
                "model_seat": int(self.model_seat),
                "seed": int(self.config.seed),
                "seat0_deck": self.seat0_deck,
                "seat1_deck": self.seat1_deck,
                "human_deck": self.config.human_deck,
                "model_deck": self.config.model_deck,
                "snapshot_registry_json": (
                    None
                    if self.config.snapshot_registry_json is None
                    else self.config.snapshot_registry_json.as_posix()
                ),
                "model_sampling_algorithm": self.config.model_sampling_algorithm,
                "search_rollout_opponent_policy_id": self.search_rollout_opponent_policy_id,
                "god_search": self.config.god_search.to_json_dict(),
            },
        )

    def _scheduled_game(self) -> ScheduledGame:
        human_proxy = self.search_rollout_opponent_policy_id
        return ScheduledGame(
            pair_index=0,
            swap_index=0,
            episode_index=0,
            episode_seed=int(self.config.seed),
            focal_policy_id=self.policy_id,
            opponent_policy_id=human_proxy,
            seat0_policy_id=self.policy_id if self.model_seat == 0 else human_proxy,
            seat1_policy_id=human_proxy if self.model_seat == 0 else self.policy_id,
            focal_seat=int(self.model_seat),
            seat0_deck=self.seat0_deck,
            seat1_deck=self.seat1_deck,
        )

    def _legal_ids_for_row(self) -> np.ndarray:
        if self.batch.ids_offsets is None:
            raise HumanPlaySessionError("current simulator batch is missing ids_offsets legality")
        legal_ids, legal_offsets = self.batch.ids_offsets
        start = int(legal_offsets[0])
        end = int(legal_offsets[1])
        return np.asarray(legal_ids[start:end], dtype=np.uint32)

    def _current_actor(self) -> int:
        return int(np.asarray(self.batch.actor, dtype=np.int64)[0])

    def _is_terminal(self) -> bool:
        return bool(np.asarray(self.batch.terminated, dtype=np.bool_)[0]) or bool(
            np.asarray(self.batch.truncated, dtype=np.bool_)[0]
        )

    def _terminal_summary(self) -> dict[str, Any]:
        result = game_result_from_step(
            self.batch,
            env_index=0,
            acting_seat=self.last_acting_seat,
            episode_seed=int(self.config.seed),
            max_decisions=getattr(self.env, "max_decisions", None),
            max_ticks=getattr(self.env, "max_ticks", None),
            max_no_progress_decisions=getattr(self.env, "max_no_progress_decisions", None),
        )
        return {
            "status": "complete",
            "terminal": True,
            "decision_count": int(self.decision_count),
            "winner_seat": result.winner_seat,
            "termination_reason": result.termination_reason,
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "engine_status": int(result.engine_status),
            "action_summary": summarize_eval_action_counters(self.action_counters),
            "god_search": self.runner.god_search_diagnostics(),
        }


def _require_human_view_api() -> Any:
    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ModuleNotFoundError as exc:
        raise HumanPlaySessionError("human play requires weiss-sim; install the sim extra first") from exc
    if not callable(getattr(weiss_sim, "human_decision_view", None)):
        raise HumanPlaySessionError(
            "weiss_sim.human_decision_view is missing. Install the simulator checkout that includes "
            "human_decision_view_v1 before launching the web play system."
        )
    return weiss_sim


def _normalize_view(view: dict[str, Any]) -> dict[str, Any]:
    legal_actions = view.get("legal_actions")
    if not isinstance(legal_actions, list):
        raise HumanPlaySessionError("human_decision_view payload is missing legal_actions[]")
    legal_ids = [int(item["action_id"]) for item in legal_actions if isinstance(item, dict) and "action_id" in item]
    payload = dict(view)
    payload["legal_action_ids"] = [int(item) for item in payload.get("legal_action_ids", legal_ids)]
    if payload["legal_action_ids"] != legal_ids:
        raise HumanPlaySessionError("human_decision_view legal_action_ids must match legal_actions order")
    return payload


def _enrich_card_labels(value: Any, weiss_sim: Any) -> Any:
    if isinstance(value, list):
        return [_enrich_card_labels(item, weiss_sim) for item in value]
    if not isinstance(value, dict):
        return value
    payload = {key: _enrich_card_labels(item, weiss_sim) for key, item in value.items()}
    if "card_id" not in payload or payload.get("name") or payload.get("card_no"):
        return payload
    cards = getattr(weiss_sim, "cards", None)
    get_card = None if cards is None else getattr(cards, "get", None)
    if not callable(get_card):
        return payload
    try:
        card = get_card(int(payload["card_id"]))
    except Exception:
        return payload
    payload.setdefault("name", str(getattr(card, "name", "") or f"Card {payload['card_id']}"))
    payload.setdefault("card_no", str(getattr(card, "card_no", "") or payload["card_id"]))
    payload.setdefault("label", f"{payload['name']} ({payload['card_no']})")
    return payload


def _repo_root_from_run_dir(run_dir: Path) -> Path:
    return run_dir.resolve().parents[1]


def _resolve_stack_config_path(run_dir: Path, stack_config: Path | None) -> Path:
    return stack_config.resolve() if stack_config is not None else (run_dir / "config_canonical.json").resolve()


def _resolve_expected_spec_hash(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        value = str(manifest.get("spec_hash256", "")).strip()
        if value:
            return value
    spec_hash_path = run_dir / "spec_hash256.txt"
    if spec_hash_path.is_file():
        value = spec_hash_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise HumanPlaySessionError(f"could not resolve spec_hash256 from {run_dir}")


def _resolve_run_id256(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        run_id256 = str(manifest.get("run_id256", "")).strip().lower()
        if len(run_id256) == 64:
            return run_id256
    return hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()


def _normalize_policy_id(run_dir: Path, requested_policy_id: str) -> str:
    policy_id = requested_policy_id.strip()
    if policy_id not in {"", "auto", "latest", "main_league_selected", "recommended"}:
        return policy_id
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    dev_eval_path = next(
        (
            candidate
            for candidate in (
                run_dir / "training" / "logs" / "dev_eval_summaries.json",
                run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if registry_path.is_file():
        registry = SnapshotRegistry.load(registry_path)
        selected_main = _selected_main_snapshot_policy_id(registry)
        if selected_main is not None and policy_id in {"", "auto", "main_league_selected", "recommended"}:
            return selected_main
        if dev_eval_path is not None:
            recommended = recommend_focal_policy_id(
                snapshot_registry=registry,
                dev_eval_summaries=load_dev_eval_summaries(dev_eval_path),
                candidate_policy_ids=[snapshot.policy_id for snapshot in registry.snapshots],
            )
            if recommended and _is_human_play_model_candidate(recommended):
                return recommended
        if registry.snapshots:
            return str(registry.snapshots[-1].policy_id)
    raise HumanPlaySessionError("could not infer a default policy id; pass policy_id explicitly")


def _selected_main_snapshot_policy_id(registry: SnapshotRegistry) -> str | None:
    for snapshot in reversed(registry.snapshots):
        policy_id = str(snapshot.policy_id).strip()
        if policy_id == "main_league_selected" or policy_id.endswith("_main_league_selected"):
            return policy_id
    return None


def _is_human_play_model_candidate(policy_id: str) -> bool:
    return str(policy_id) not in {
        RANDOM_LEGAL_POLICY_ID,
        NO_LEAGUE_POLICY_ID,
        LEGACY_NO_LEAGUE_POLICY_ID,
        HEURISTIC_PUBLIC_POLICY_ID,
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    }


def _deck_for_seat(*, config: HumanPlayConfig, seat: int) -> str:
    if int(seat) == int(config.human_seat):
        return config.human_deck
    return config.model_deck or deck_id_for_policy_id(config.policy_id)


def _label_for_action(view: dict[str, Any], action_id: int) -> str:
    for item in view.get("legal_actions", []):
        if isinstance(item, dict) and int(item.get("action_id", -1)) == int(action_id):
            return str(item.get("label") or item.get("short_label") or f"Action {action_id}")
    return f"Action {action_id}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HumanPlaySessionError(f"{path} must contain a JSON object")
    return payload


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
