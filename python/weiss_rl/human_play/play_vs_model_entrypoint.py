from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.config import load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog, DecodedAction
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval import load_dev_eval_summaries, sample_action_pinned
from weiss_rl.eval.policies.set import recommend_focal_policy_id
from weiss_rl.eval.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.simulator_runner import ResolvedEvalPolicy, resolve_eval_policies
from weiss_rl.league.registry import SnapshotRegistry

weiss_sim: ModuleType | None
try:
    import weiss_sim as _weiss_sim
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in non-sim test environments
    weiss_sim = None
else:
    weiss_sim = _weiss_sim

_SLOT_NAMES = ("front_left", "front_center", "front_right", "back_left", "back_right")


@dataclass(frozen=True, slots=True)
class _ActionOption:
    index: int
    action_id: int
    label: str
    probability: float | None = None


def _require_weiss_sim() -> Any:
    if weiss_sim is None:
        raise RuntimeError(
            "play_vs_model.py requires the optional weiss-sim dependency. Install with `uv sync --extra dev --extra sim` "
            "or `python -m pip install -e '.[dev,sim]'`."
        )
    return weiss_sim


def _repo_root_from_run_dir(run_dir: Path) -> Path:
    return run_dir.resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_stack_config_path(run_dir: Path, stack_config: Path | None) -> Path:
    if stack_config is not None:
        return stack_config.resolve()
    return (run_dir / "config_canonical.json").resolve()


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
    raise ValueError(f"could not resolve spec_hash256 from {run_dir}")


def _normalize_policy_id(run_dir: Path, requested_policy_id: str) -> str:
    policy_id = requested_policy_id.strip()
    if policy_id:
        return policy_id

    final_eval_summary = run_dir / "eval" / "final_eval" / "summary.json"
    if final_eval_summary.is_file():
        summary = _load_json(final_eval_summary)
        metadata = summary.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("recommended_focal_policy_id", "focal_policy_id"):
                value = str(metadata.get(key, "")).strip()
                if value:
                    return value
            focal_policy = metadata.get("focal_policy", {})
            if isinstance(focal_policy, dict):
                value = str(focal_policy.get("policy_id", "")).strip()
                if value:
                    return value

    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    dev_eval_path = None
    for candidate in (
        run_dir / "training" / "logs" / "dev_eval_summaries.json",
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
    ):
        if candidate.is_file():
            dev_eval_path = candidate
            break
    if registry_path.is_file():
        registry = SnapshotRegistry.load(registry_path)
        if dev_eval_path is not None:
            recommended = recommend_focal_policy_id(
                snapshot_registry=registry,
                dev_eval_summaries=load_dev_eval_summaries(dev_eval_path),
                candidate_policy_ids=[snapshot.policy_id for snapshot in registry.snapshots],
            )
            if recommended:
                return recommended
        if registry.snapshots:
            return str(registry.snapshots[-1].policy_id)
    raise ValueError("could not infer a default policy id; pass --policy-id explicitly or finalize the run first")


def _legal_ids_for_row(batch: DecisionBoundaryBatch, env_index: int = 0) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("human play requires ids_offsets legality")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    return np.asarray(legal_ids[start:end], dtype=np.int64)


def _slot_name(slot: int | None) -> str:
    if slot is None:
        return "?"
    if 0 <= int(slot) < len(_SLOT_NAMES):
        return _SLOT_NAMES[int(slot)]
    return str(int(slot))


def _format_decoded_action(action_id: int, catalog: ActionCatalog) -> str:
    try:
        decoded = catalog.decode(int(action_id))
        return _format_catalog_action(decoded)
    except Exception:
        sim = _require_weiss_sim()
        raw = sim.decode_action_id(int(action_id))
        if not isinstance(raw, dict):
            return str(raw)
        family = str(raw.get("family", "unknown"))
        params = raw.get("params", [])
        return f"{family} {params}"


def _format_catalog_action(decoded: DecodedAction) -> str:
    family = decoded.family
    if family == "main_play_character":
        return f"{family} hand={decoded.hand_index} -> {_slot_name(decoded.stage_slot)}"
    if family == "main_move":
        return f"{family} {_slot_name(decoded.from_slot)} -> {_slot_name(decoded.to_slot)}"
    if family == "attack":
        return f"{family} {_slot_name(decoded.slot)} type={decoded.attack_type}"
    if family in {"clock_from_hand", "main_play_event", "climax_play", "mulligan_select"}:
        return f"{family} hand={decoded.hand_index}"
    if family in {"level_up", "trigger_order", "choice_select"}:
        return f"{family} index={decoded.index}"
    if family in {"encore_pay", "encore_decline"}:
        return f"{family} slot={_slot_name(decoded.slot)}"
    return family


def _rank_legal_actions(
    *,
    logits: np.ndarray,
    legal_ids: np.ndarray,
    catalog: ActionCatalog,
    top_k: int,
) -> list[_ActionOption]:
    if legal_ids.size == 0:
        return []
    legal_logits = np.asarray(logits[legal_ids], dtype=np.float64)
    shifted = legal_logits - np.max(legal_logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    order = np.argsort(probs)[::-1][: max(1, int(top_k))]
    return [
        _ActionOption(
            index=int(rank),
            action_id=int(legal_ids[idx]),
            label=_format_decoded_action(int(legal_ids[idx]), catalog),
            probability=float(probs[idx]),
        )
        for rank, idx in enumerate(order, start=1)
    ]


def _choose_policy_action(
    *,
    policy: ResolvedEvalPolicy,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    pass_action_id: int,
    seat_hidden: torch.Tensor | None,
    rng: Pcg32XshRrV1,
    temperature: float,
    top_k: int,
    catalog: ActionCatalog,
) -> tuple[int, torch.Tensor | None, list[_ActionOption]]:
    if policy.heuristic_policy is not None:
        action = int(policy.heuristic_policy.choose_action(np.asarray(batch.obs[0]), legal_ids))
        return action, seat_hidden, []
    if policy.kind == "random_legal":
        if legal_ids.size == 0:
            return int(pass_action_id), seat_hidden, []
        sample = int(legal_ids[int(rng.next_u64() % legal_ids.size)])
        return sample, seat_hidden, []
    logits, next_hidden = _model_logits_for_state(policy=policy, batch=batch, seat_hidden=seat_hidden)
    ranked = _rank_legal_actions(logits=logits, legal_ids=legal_ids, catalog=catalog, top_k=top_k)
    if float(temperature) <= 0.0:
        if ranked:
            return int(ranked[0].action_id), next_hidden, ranked
        return int(pass_action_id), next_hidden, ranked
    sample_logits = logits if float(temperature) == 1.0 else np.asarray(logits / float(temperature), dtype=np.float32)
    action, _logp = sample_action_pinned(
        sample_logits,
        legal_ids,
        rng=rng,
        pass_action_id=pass_action_id,
    )
    return int(action), next_hidden, ranked


def _model_logits_for_state(
    *,
    policy: ResolvedEvalPolicy,
    batch: DecisionBoundaryBatch,
    seat_hidden: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    model = policy.model
    if model is None:
        raise RuntimeError(f"policy {policy.policy_id!r} does not expose a playable model")

    device = torch.device("cpu")
    obs_tensor = torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=device)
    actor_tensor = torch.as_tensor(np.asarray(batch.actor, dtype=np.int64), device=device)
    assert seat_hidden is not None
    with torch.inference_mode():
        logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
            obs_tensor,
            actor_tensor,
            seat_hidden,
        )
    return np.asarray(logits_tensor.detach().cpu().numpy()[0], dtype=np.float32), next_hidden


def _rank_policy_hint_options(
    *,
    policy: ResolvedEvalPolicy,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    seat_hidden: torch.Tensor | None,
    top_k: int,
    catalog: ActionCatalog,
) -> tuple[list[_ActionOption], torch.Tensor | None]:
    if policy.model is None:
        return [], seat_hidden
    logits, next_hidden = _model_logits_for_state(policy=policy, batch=batch, seat_hidden=seat_hidden)
    return _rank_legal_actions(logits=logits, legal_ids=legal_ids, catalog=catalog, top_k=top_k), next_hidden


def _print_deck_list() -> None:
    sim = _require_weiss_sim()
    names = sim.cards.presets()
    print("Available preset decks:")
    for name in names:
        details = cast(dict[str, Any], sim.cards.describe_deck(name, rules_profile="approx", card_pool="all"))
        cards = cast(list[dict[str, Any]], details.get("cards", []))
        counts = cast(list[dict[str, Any]], details.get("counts", []))
        unique_cards = len({int(card["id"]) for card in cards})
        first_cards = ", ".join(str(card["card_no"]) for card in counts[:4])
        print(f"- {name}: 50 cards, {unique_cards} unique cards, sample {first_cards}")


def _clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def _render_board(env: DecisionBoundaryEnv, *, perspective: int) -> str:
    render_ansi = getattr(env.pool, "render_ansi", None)
    if not callable(render_ansi):
        raise RuntimeError("simulator runtime does not expose render_ansi")
    return str(render_ansi(0, int(perspective)))


def _print_model_suggestions(options: list[_ActionOption], *, header: str) -> None:
    if not options:
        return
    print(header)
    for option in options:
        prob = "" if option.probability is None else f" ({option.probability:.3f})"
        print(f"  [{option.index}] {option.label}{prob}")


def _read_console_input(*, prompt: str, eof_message: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise KeyboardInterrupt(eof_message) from exc


def _prompt_human_action(
    *, batch: DecisionBoundaryBatch, catalog: ActionCatalog, top_k_hints: list[_ActionOption]
) -> int:
    legal_ids = _legal_ids_for_row(batch)
    options = [
        _ActionOption(index=index, action_id=int(action_id), label=_format_decoded_action(int(action_id), catalog))
        for index, action_id in enumerate(legal_ids.tolist(), start=1)
    ]
    if not sys.stdin.isatty():
        raise KeyboardInterrupt("stdin is not interactive; launch play_vs_model.py in a TTY to control the human seat")

    while True:
        print("Your legal actions:")
        for option in options:
            print(f"  [{option.index}] {option.label}")
        if top_k_hints:
            _print_model_suggestions(top_k_hints, header="Model hints for this state:")
        raw = (
            _read_console_input(
                prompt="Choose action number, 'h' for hints, or 'q' to quit: ",
                eof_message="stdin closed while waiting for human input; launch play_vs_model.py in a TTY to play",
            )
            .strip()
            .lower()
        )
        if raw == "q":
            raise KeyboardInterrupt("user quit human-play session")
        if raw == "h":
            continue
        if raw.isdigit():
            choice = int(raw)
            for option in options:
                if option.index == choice:
                    return int(option.action_id)
        print("Invalid choice.")


def _advance_after_model_action(*, env: DecisionBoundaryEnv, action: int) -> DecisionBoundaryBatch:
    if sys.stdin.isatty():
        try:
            input("Press Enter to continue...")
        except EOFError:
            pass
    return env.step(np.asarray([action], dtype=np.uint32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a human-vs-model Weiss Schwarz game in the terminal")
    parser.add_argument("--run-dir", type=Path, required=False, help="Training run directory with snapshots")
    parser.add_argument(
        "--stack-config", type=Path, default=None, help="Optional stack config; defaults to run config_canonical.json"
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="",
        help="Policy id to load; defaults to the finalized focal policy when available",
    )
    parser.add_argument("--human-seat", type=int, default=0, choices=(0, 1), help="Seat controlled by the human")
    parser.add_argument("--seed", type=int, default=20260419, help="Episode seed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model action temperature; 0.0 is greedy")
    parser.add_argument("--top-k", type=int, default=5, help="How many model suggestions to show")
    parser.add_argument(
        "--deck", type=str, default="", help="Optional deck override, for example preset:main_deck_5hy_yotsuba_v1"
    )
    parser.add_argument("--opponent-deck", type=str, default="", help="Optional opponent deck override")
    parser.add_argument("--list-decks", action="store_true", help="Print bundled deck presets and exit")
    args = parser.parse_args()

    if args.list_decks:
        _print_deck_list()
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --list-decks is used")

    run_dir = args.run_dir.resolve()
    stack_config_path = _resolve_stack_config_path(run_dir, args.stack_config)
    stack = load_stack_config(stack_config_path)
    contract = load_verified_simulator_contract(
        _repo_root_from_run_dir(run_dir),
        expected_spec_hash=_resolve_expected_spec_hash(run_dir),
    )
    spec_bundle = contract.spec_bundle
    observation_dim = int(spec_bundle["observation"]["obs_len"])
    action_dim = int(spec_bundle["action"]["action_space_size"])
    pass_action_id = int(spec_bundle["action"]["pass_action_id"])
    catalog = ActionCatalog.from_spec_bundle(spec_bundle)

    policy_id = _normalize_policy_id(run_dir, str(args.policy_id))
    policies = resolve_eval_policies(
        stack=stack,
        policy_ids=[policy_id],
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=spec_bundle,
    )
    policy = policies[policy_id]
    if policy.model is not None:
        policy.model.eval()

    env_config = build_env_config_from_stack(
        stack,
        seed=int(args.seed),
        deck=(str(args.deck).strip() or None),
        opponent_deck=(str(args.opponent_deck).strip() or None),
    )
    pool, layout_name = make_env_pool_from_config(env_config, profile="fast", num_envs=1)
    if layout_name != "i16_legal_ids":
        raise RuntimeError(f"human play requires ids legality; got layout {layout_name!r}")
    env = DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
    )
    rng = Pcg32XshRrV1(int(args.seed) ^ 0xD1CEFACE)
    model_hidden: torch.Tensor | None = None
    if policy.model is not None:
        model_hidden = policy.model.initial_seat_hidden(1, device=torch.device("cpu"))

    print(f"Loaded policy {policy_id}")
    if "deck" in env_config:
        print(f"Human deck: {env_config['deck']}")
    if "opponent_deck" in env_config:
        print(f"Opponent deck: {env_config['opponent_deck']}")

    batch = env.reset(seed=int(args.seed))
    try:
        while True:
            _clear_screen()
            print(_render_board(env, perspective=int(args.human_seat)))
            if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                reward = float(np.asarray(batch.reward, dtype=np.float32)[0])
                if batch.terminated[0]:
                    outcome = "terminated"
                elif batch.truncated[0]:
                    outcome = "truncated"
                else:
                    outcome = "finished"
                print(f"\nGame {outcome}. Final reward from acting-seat view: {reward:.3f}")
                break

            current_seat = int(np.asarray(batch.actor, dtype=np.int64)[0])
            legal_ids = _legal_ids_for_row(batch)
            hint_options: list[_ActionOption] = []
            if current_seat == int(args.human_seat) and policy.model is not None:
                hint_options, next_hidden = _rank_policy_hint_options(
                    policy=policy,
                    batch=batch,
                    legal_ids=legal_ids,
                    seat_hidden=model_hidden,
                    top_k=int(args.top_k),
                    catalog=catalog,
                )
                model_hidden = next_hidden

            if current_seat == int(args.human_seat):
                action = _prompt_human_action(batch=batch, catalog=catalog, top_k_hints=hint_options)
                print(f"You chose: {_format_decoded_action(action, catalog)}")
                batch = env.step(np.asarray([action], dtype=np.uint32))
                continue

            action, model_hidden, ranked = _choose_policy_action(
                policy=policy,
                batch=batch,
                legal_ids=legal_ids,
                pass_action_id=pass_action_id,
                seat_hidden=model_hidden,
                rng=rng,
                temperature=max(float(args.temperature), 0.0),
                top_k=int(args.top_k),
                catalog=catalog,
            )
            _print_model_suggestions(ranked, header="\nModel suggestions:")
            print(f"\nModel chose: {_format_decoded_action(action, catalog)}")
            batch = _advance_after_model_action(env=env, action=action)
    except KeyboardInterrupt as exc:
        print(f"\nSession ended: {exc}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
