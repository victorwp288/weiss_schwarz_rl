"""Public-safe toy catalog and demo-only end-to-end evaluation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import parse_seed_file, stable_hash64
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.core.spec import spec_bundle_hash
from weiss_rl.eval import GameResult, ScheduledGame, run_final_eval
from weiss_rl.eval.policies.set import NO_LEAGUE_POLICY_ID, RANDOM_LEGAL_POLICY_ID

PUBLIC_DEMO_MODE = "public_demo"
PUBLIC_DEMO_WARNING = (
    "Toy public-demo artifacts are synthetic smoke artifacts only and must not be cited as thesis results."
)
PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS = 8
PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES = 64
PUBLIC_DEMO_CATALOG_FILENAME = "catalog.json"
PUBLIC_DEMO_POLICY_MANIFEST_FILENAME = "policy_manifest.json"
PUBLIC_DEMO_NOTE_FILENAME = "README.txt"
_PUBLIC_DEMO_U64_DENOMINATOR = float(1 << 64)

_PUBLIC_DEMO_SPEC_BUNDLE: dict[str, Any] = {
    "policy_version": 1,
    "spec_hash": 9001001,
    "observation": {
        "obs_encoding_version": 2,
        "dtype": "i32",
        "obs_len": 512,
    },
    "action": {
        "action_encoding_version": 1,
        "action_space_size": 9,
        "pass_action_id": 8,
    },
}

_PUBLIC_DEMO_CARD_POOL: tuple[dict[str, Any], ...] = (
    {"card_id": 1001, "feature_slot": 0, "level": 0, "power_band": 1, "soul": 1, "tag_code": 11},
    {"card_id": 1002, "feature_slot": 1, "level": 0, "power_band": 2, "soul": 1, "tag_code": 11},
    {"card_id": 1003, "feature_slot": 2, "level": 1, "power_band": 3, "soul": 1, "tag_code": 17},
    {"card_id": 1004, "feature_slot": 3, "level": 1, "power_band": 2, "soul": 2, "tag_code": 17},
    {"card_id": 1005, "feature_slot": 4, "level": 2, "power_band": 4, "soul": 2, "tag_code": 23},
    {"card_id": 1006, "feature_slot": 5, "level": 2, "power_band": 3, "soul": 1, "tag_code": 23},
    {"card_id": 1007, "feature_slot": 6, "level": 0, "power_band": 1, "soul": 1, "tag_code": 31},
    {"card_id": 1008, "feature_slot": 7, "level": 1, "power_band": 2, "soul": 1, "tag_code": 31},
    {"card_id": 1009, "feature_slot": 8, "level": 2, "power_band": 4, "soul": 2, "tag_code": 37},
    {"card_id": 1010, "feature_slot": 9, "level": 1, "power_band": 3, "soul": 1, "tag_code": 37},
    {"card_id": 1011, "feature_slot": 10, "level": 0, "power_band": 2, "soul": 1, "tag_code": 41},
    {"card_id": 1012, "feature_slot": 11, "level": 2, "power_band": 4, "soul": 2, "tag_code": 41},
)

_PUBLIC_DEMO_DECKS: tuple[dict[str, Any], ...] = (
    {
        "deck_id": "toy_deck_tempo_v1",
        "display_name": "Toy Tempo Deck",
        "card_ids": [1001, 1002, 1003, 1004, 1007, 1008],
        "counts": [4, 4, 4, 4, 2, 2],
        "mechanical_profile": {"pressure": 0.76, "resilience": 0.32, "curve": 0.71},
    },
    {
        "deck_id": "toy_deck_guard_v1",
        "display_name": "Toy Guard Deck",
        "card_ids": [1003, 1004, 1005, 1006, 1011, 1012],
        "counts": [4, 4, 4, 4, 2, 2],
        "mechanical_profile": {"pressure": 0.41, "resilience": 0.84, "curve": 0.43},
    },
    {
        "deck_id": "toy_deck_midrange_v1",
        "display_name": "Toy Midrange Deck",
        "card_ids": [1001, 1004, 1006, 1008, 1009, 1010],
        "counts": [4, 4, 4, 4, 2, 2],
        "mechanical_profile": {"pressure": 0.58, "resilience": 0.63, "curve": 0.59},
    },
)


@dataclass(frozen=True, slots=True)
class ToyPublicPolicy:
    policy_id: str
    deck_id: str
    aggression: float
    control: float
    discipline: float
    volatility: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "deck_id": self.deck_id,
            "aggression": self.aggression,
            "control": self.control,
            "discipline": self.discipline,
            "volatility": self.volatility,
        }


_PUBLIC_DEMO_POLICIES: tuple[ToyPublicPolicy, ...] = (
    ToyPublicPolicy(
        policy_id=RANDOM_LEGAL_POLICY_ID,
        deck_id="toy_deck_tempo_v1",
        aggression=0.44,
        control=0.44,
        discipline=0.45,
        volatility=0.52,
    ),
    ToyPublicPolicy(
        policy_id=NO_LEAGUE_POLICY_ID,
        deck_id="toy_deck_guard_v1",
        aggression=0.49,
        control=0.57,
        discipline=0.61,
        volatility=0.36,
    ),
    ToyPublicPolicy(
        policy_id="toy_policy_000100",
        deck_id="toy_deck_midrange_v1",
        aggression=0.61,
        control=0.58,
        discipline=0.68,
        volatility=0.29,
    ),
    ToyPublicPolicy(
        policy_id="toy_policy_000200",
        deck_id="toy_deck_guard_v1",
        aggression=0.53,
        control=0.73,
        discipline=0.81,
        volatility=0.18,
    ),
)


@dataclass(frozen=True, slots=True)
class PublicDemoArtifacts:
    catalog_path: Path
    policy_manifest_path: Path
    note_path: Path
    scalars_path: Path
    bundle_path: Path
    policy_ids: tuple[str, ...]
    recommended_focal_policy_id: str


class _ToyPublicDemoRunner:
    def __init__(self, policies: dict[str, ToyPublicPolicy], decks: dict[str, dict[str, Any]]) -> None:
        self._policies = policies
        self._decks = decks

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        seat0 = self._policies[scheduled_game.seat0_policy_id]
        seat1 = self._policies[scheduled_game.seat1_policy_id]
        score0 = self._seat_score(seat0, episode_seed=scheduled_game.episode_seed, seat=0)
        score1 = self._seat_score(seat1, episode_seed=scheduled_game.episode_seed, seat=1)
        diff = score0 - score1
        closeness = max(0.0, 1.0 - min(abs(diff), 1.0))
        draw_prob = 0.04 + 0.12 * closeness
        trunc_prob = 0.01 + 0.02 * max(seat0.volatility, seat1.volatility)
        total_mass = draw_prob + trunc_prob
        outcome_draw = self._unit_float(
            "outcome",
            scheduled_game.episode_seed,
            scheduled_game.swap_index,
            scheduled_game.seat0_policy_id,
            scheduled_game.seat1_policy_id,
        )
        if outcome_draw < trunc_prob:
            return GameResult(
                episode_seed=scheduled_game.episode_seed,
                terminated=False,
                truncated=True,
                winner_seat=None,
            )
        if outcome_draw < total_mass:
            return GameResult(
                episode_seed=scheduled_game.episode_seed,
                terminated=True,
                truncated=False,
                winner_seat=None,
            )

        win_mass = max(1e-6, 1.0 - total_mass)
        seat0_win_prob = self._sigmoid(diff * 4.0)
        normalized = (outcome_draw - total_mass) / win_mass
        winner_seat = 0 if normalized < seat0_win_prob else 1
        return GameResult(
            episode_seed=scheduled_game.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=winner_seat,
        )

    def _seat_score(self, policy: ToyPublicPolicy, *, episode_seed: int, seat: int) -> float:
        deck = self._decks[policy.deck_id]
        profile = deck["mechanical_profile"]
        tempo = self._unit_float("tempo", episode_seed)
        stability = self._unit_float("stability", episode_seed)
        pressure = self._unit_float("pressure", episode_seed)
        seat_bias = 0.012 if seat == 0 else -0.012
        return (
            0.34 * policy.aggression * (0.55 + tempo)
            + 0.28 * policy.control * (0.60 + stability)
            + 0.18 * policy.discipline * (0.65 + stability)
            + 0.10 * float(profile["pressure"]) * (0.50 + pressure)
            + 0.06 * float(profile["resilience"]) * (0.55 + stability)
            + 0.04 * float(profile["curve"]) * (0.50 + tempo)
            - 0.08 * policy.volatility * (0.40 + pressure)
            + seat_bias
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _unit_float(*parts: object) -> float:
        payload = "|".join(str(part) for part in parts).encode("utf-8")
        return stable_hash64(payload) / _PUBLIC_DEMO_U64_DENOMINATOR


def public_demo_spec_bundle() -> dict[str, Any]:
    return json.loads(json.dumps(_PUBLIC_DEMO_SPEC_BUNDLE))


def public_demo_spec_hash256() -> str:
    return spec_bundle_hash(_PUBLIC_DEMO_SPEC_BUNDLE)


def public_demo_simulator_info() -> dict[str, Any]:
    return {
        "build_info": "builtin-public-demo-v1",
        "db_info": "synthetic-catalog-only",
        "compatibility_hash": str(_PUBLIC_DEMO_SPEC_BUNDLE["spec_hash"]),
        "public_safe": True,
        "runtime": PUBLIC_DEMO_MODE,
    }


def public_demo_policy_ids() -> tuple[str, ...]:
    return tuple(policy.policy_id for policy in _PUBLIC_DEMO_POLICIES)


def public_demo_default_focal_policy_id() -> str:
    return _PUBLIC_DEMO_POLICIES[-1].policy_id


def public_demo_stop_rules() -> StopRulesConfig:
    return StopRulesConfig(stop_delta_ci_half_width=0.08, stop_confidence=0.95)


def stage_public_demo_run(run_dir: Path) -> PublicDemoArtifacts:
    public_demo_dir = run_dir / "public_demo"
    public_demo_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = public_demo_dir / PUBLIC_DEMO_CATALOG_FILENAME
    policy_manifest_path = public_demo_dir / PUBLIC_DEMO_POLICY_MANIFEST_FILENAME
    note_path = public_demo_dir / PUBLIC_DEMO_NOTE_FILENAME

    catalog_payload = {
        "kind": "toy_public_demo_catalog_v1",
        "mode": PUBLIC_DEMO_MODE,
        "public_safe": True,
        "warning": PUBLIC_DEMO_WARNING,
        "spec_bundle": public_demo_spec_bundle(),
        "card_pool": list(_PUBLIC_DEMO_CARD_POOL),
        "decks": list(_PUBLIC_DEMO_DECKS),
    }
    policy_manifest_payload = {
        "kind": "toy_public_demo_policy_manifest_v1",
        "mode": PUBLIC_DEMO_MODE,
        "public_safe": True,
        "warning": PUBLIC_DEMO_WARNING,
        "recommended_focal_policy_id": public_demo_default_focal_policy_id(),
        "policy_ids": list(public_demo_policy_ids()),
        "policies": [policy.to_dict() for policy in _PUBLIC_DEMO_POLICIES],
    }

    catalog_path.write_text(json.dumps(catalog_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    policy_manifest_path.write_text(
        json.dumps(policy_manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note_path.write_text(
        "\n".join(
            (
                "Public demo mode stages a synthetic card catalog and deterministic toy policies.",
                "It does not execute proprietary simulator content or claim thesis-grade performance.",
                PUBLIC_DEMO_WARNING,
                "",
            )
        ),
        encoding="utf-8",
    )

    training_dir = run_dir / "training"
    logs_dir = training_dir / "logs"
    checkpoints_dir = training_dir / "checkpoints"
    for path in (training_dir, logs_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)

    scalars_path = logs_dir / "scalars.jsonl"
    scalars_record = {
        "mode": PUBLIC_DEMO_MODE,
        "toy_public_demo": True,
        "warning": PUBLIC_DEMO_WARNING,
        "staged_policy_count": len(_PUBLIC_DEMO_POLICIES),
        "staged_deck_count": len(_PUBLIC_DEMO_DECKS),
        "staged_card_count": len(_PUBLIC_DEMO_CARD_POOL),
    }
    scalars_path.write_text(json.dumps(scalars_record, sort_keys=True) + "\n", encoding="utf-8")

    bundle_path = checkpoints_dir / "toy_policy_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "kind": "toy_public_demo_checkpoint_bundle_v1",
                "mode": PUBLIC_DEMO_MODE,
                "warning": PUBLIC_DEMO_WARNING,
                "catalog_path": catalog_path.relative_to(run_dir).as_posix(),
                "policy_manifest_path": policy_manifest_path.relative_to(run_dir).as_posix(),
                "policy_ids": list(public_demo_policy_ids()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return PublicDemoArtifacts(
        catalog_path=catalog_path,
        policy_manifest_path=policy_manifest_path,
        note_path=note_path,
        scalars_path=scalars_path,
        bundle_path=bundle_path,
        policy_ids=public_demo_policy_ids(),
        recommended_focal_policy_id=public_demo_default_focal_policy_id(),
    )


def run_public_demo_final_eval(
    *,
    output_dir: Path,
    run_dir: Path,
    paired_seed_file: Path,
    paired_seed_limit: int = PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
    sample_count: int = PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    stop_rules: StopRulesConfig | None = None,
) -> dict[str, Any]:
    if paired_seed_limit < 1:
        raise ValueError("paired_seed_limit must be >= 1")
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    public_demo_dir = run_dir / "public_demo"
    catalog_path = public_demo_dir / PUBLIC_DEMO_CATALOG_FILENAME
    policy_manifest_path = public_demo_dir / PUBLIC_DEMO_POLICY_MANIFEST_FILENAME
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing staged public-demo catalog: {catalog_path}")
    if not policy_manifest_path.is_file():
        raise FileNotFoundError(f"missing staged public-demo policy manifest: {policy_manifest_path}")

    policy_manifest = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    raw_policies = policy_manifest.get("policies", [])
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("public-demo policy manifest must contain a non-empty policies list")

    policies = {
        str(item["policy_id"]): ToyPublicPolicy(
            policy_id=str(item["policy_id"]),
            deck_id=str(item["deck_id"]),
            aggression=float(item["aggression"]),
            control=float(item["control"]),
            discipline=float(item["discipline"]),
            volatility=float(item["volatility"]),
        )
        for item in raw_policies
    }
    decks = {str(deck["deck_id"]): deck for deck in _PUBLIC_DEMO_DECKS}
    policy_ids = tuple(str(policy_id) for policy_id in policy_manifest.get("policy_ids", []))
    if not policy_ids:
        policy_ids = tuple(policies)

    paired_seeds = tuple(parse_seed_file(paired_seed_file)[:paired_seed_limit])
    if len(paired_seeds) < paired_seed_limit:
        raise ValueError(
            "public-demo eval requested "
            f"{paired_seed_limit} paired seeds but only found {len(paired_seeds)} "
            f"in {paired_seed_file}"
        )

    effective_stop_rules = stop_rules or public_demo_stop_rules()
    stage1_paired_seeds = min(4, paired_seed_limit)
    metadata = {
        "mode": PUBLIC_DEMO_MODE,
        "public_safe": True,
        "demo_only": True,
        "warning": PUBLIC_DEMO_WARNING,
        "source_run_dir": run_dir.as_posix(),
        "catalog_path": catalog_path.relative_to(run_dir).as_posix(),
        "policy_manifest_path": policy_manifest_path.relative_to(run_dir).as_posix(),
        "recommended_focal_policy_id": str(policy_manifest.get("recommended_focal_policy_id", "")),
    }
    runner = _ToyPublicDemoRunner(policies=policies, decks=decks)
    return run_final_eval(
        output_dir=output_dir,
        runner=runner,
        policy_ids=policy_ids,
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=paired_seed_limit,
        stop_rules=effective_stop_rules,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        sample_count=sample_count,
        metadata=metadata,
        seed_file_path=paired_seed_file,
    )
