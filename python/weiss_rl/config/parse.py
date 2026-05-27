"""Strict YAML parsing and grouped preset loading for the RL config."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from .models import (
    CurriculumConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LeagueConfig,
    LockedConfig,
    ModelConfig,
    ReproducibilityConfig,
    RewardsConfig,
    StackConfig,
    SystemConfig,
    TrainingConfig,
)
from .parsing_utils import (
    PRESET_TOP_LEVEL_KEYS,
    deep_merge,
    load_json,
    load_preset_document,
    load_yaml,
    reject_unknown_keys,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_int_list,
    require_mapping,
    require_str_list,
    require_text,
    resolve_repo_path,
    resolve_repo_root,
)
from .sections_core import EXPERIMENT_ROLES, parse_experiment_config, parse_system_config
from .sections_curriculum import normalize_curriculum_payload, parse_curriculum_config
from .sections_environment import parse_environment_config, parse_rewards_config
from .sections_evaluation import parse_evaluation_config
from .sections_league import parse_league_config
from .sections_model import (
    MODEL_ENCODER_KINDS,
    MODEL_RECURRENT_CORES,
    STRUCTURED_POLICY_CONTRACTS,
    parse_model_config,
)
from .sections_reproducibility import parse_reproducibility_config
from .sections_training import (
    TRAINING_ACTOR_POLICY_BACKENDS,
    TRAINING_ALGORITHMS,
    TRAINING_FIXED_OPPONENT_BACKENDS,
    TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
    TRAINING_PUBLIC_HEURISTIC_PROFILES,
    TRAINING_STRUCTURED_METRICS_MODES,
    TRAINING_TEACHER_AUX_MODES,
    parse_training_config,
)
from .seed_sets import parse_seed_sets_override, resolve_seed_sets

_EXPERIMENT_ROLES = EXPERIMENT_ROLES
_MODEL_ENCODER_KINDS = MODEL_ENCODER_KINDS
_STRUCTURED_POLICY_CONTRACTS = STRUCTURED_POLICY_CONTRACTS
_MODEL_RECURRENT_CORES = MODEL_RECURRENT_CORES
_TRAINING_ALGORITHMS = TRAINING_ALGORITHMS
_TRAINING_STRUCTURED_METRICS_MODES = TRAINING_STRUCTURED_METRICS_MODES
_TRAINING_TEACHER_AUX_MODES = TRAINING_TEACHER_AUX_MODES
_TRAINING_FIXED_OPPONENT_BACKENDS = TRAINING_FIXED_OPPONENT_BACKENDS
_TRAINING_ACTOR_POLICY_BACKENDS = TRAINING_ACTOR_POLICY_BACKENDS
_TRAINING_PUBLIC_HEURISTIC_PROFILES = TRAINING_PUBLIC_HEURISTIC_PROFILES
_TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES = TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES
_TOP_LEVEL_KEYS = PRESET_TOP_LEVEL_KEYS
_CANONICAL_CONFIG_KEYS = frozenset({"schema_version", "description", "config", "seed_sets"})
_CONFIG_SECTION_KEYS = frozenset(
    {
        "experiment",
        "system",
        "model",
        "training",
        "environment",
        "rewards",
        "curriculum",
        "league",
        "evaluation",
        "reproducibility",
    }
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def _load_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def _resolve_repo_root(stack_file: Path) -> Path:
    return resolve_repo_root(stack_file)


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    return require_mapping(value, context=context)


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    return require_int(value, field_name=field_name, minimum=minimum)


def _require_float(value: Any, *, field_name: str) -> float:
    return require_float(value, field_name=field_name)


def _require_bool(value: Any, *, field_name: str) -> bool:
    return require_bool(value, field_name=field_name)


def _require_text(value: Any, *, field_name: str) -> str:
    return require_text(value, field_name=field_name)


def _require_choice(value: Any, *, field_name: str, allowed: Collection[str]) -> str:
    return require_choice(value, field_name=field_name, allowed=allowed)


def _require_str_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    return require_str_list(value, field_name=field_name)


def _require_int_list(value: Any, *, field_name: str) -> tuple[int, ...]:
    return require_int_list(value, field_name=field_name)


def _reject_unknown_keys(body: Mapping[str, Any], *, allowed: Collection[str], context: str) -> None:
    reject_unknown_keys(body, allowed=allowed, context=context)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(base, overlay)


def _load_preset_document(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    return load_preset_document(path, seen=seen)


def _resolve_repo_path(root: Path, raw_path: str) -> Path:
    return resolve_repo_path(root, raw_path)


def _parse_experiment_config(body: dict[str, Any]) -> ExperimentConfig:
    return parse_experiment_config(body)


def _parse_system_config(body: dict[str, Any]) -> SystemConfig:
    return parse_system_config(body)


def _parse_model_config(body: dict[str, Any]) -> ModelConfig:
    return parse_model_config(body)


def _parse_training_config(body: dict[str, Any]) -> TrainingConfig:
    return parse_training_config(body)


def _parse_environment_config(body: dict[str, Any]) -> EnvironmentConfig:
    return parse_environment_config(body)


def _parse_rewards_config(body: dict[str, Any]) -> RewardsConfig:
    return parse_rewards_config(body)


def _normalize_curriculum_payload(value: Any, *, field_name: str) -> Any:
    return normalize_curriculum_payload(value, field_name=field_name)


def _parse_curriculum_config(body: dict[str, Any] | None) -> CurriculumConfig:
    return parse_curriculum_config(body)


def _parse_league_config(body: dict[str, Any]) -> LeagueConfig:
    return parse_league_config(body)


def _parse_evaluation_config(body: dict[str, Any]) -> EvaluationConfig:
    return parse_evaluation_config(body)


def _parse_reproducibility_config(body: dict[str, Any]) -> ReproducibilityConfig:
    return parse_reproducibility_config(body)


def _resolve_seed_sets(
    *,
    root: Path,
    league: LeagueConfig | None,
    evaluation: EvaluationConfig | None,
    reproducibility: ReproducibilityConfig | None,
) -> dict[str, Path]:
    return resolve_seed_sets(root=root, league=league, evaluation=evaluation, reproducibility=reproducibility)


def _parse_seed_sets_override(*, root: Path, seed_sets_doc: Mapping[str, Any]) -> dict[str, Path]:
    return parse_seed_sets_override(root=root, seed_sets_doc=seed_sets_doc)


def _build_stack_config_from_component_doc(
    *,
    root: Path,
    component_root_doc: Mapping[str, Any],
    description: str,
    schema_version: int | None,
    seed_sets_override: Mapping[str, Any] | None = None,
    lock_intent: dict[str, Any] | None = None,
) -> StackConfig:
    _reject_unknown_keys(component_root_doc, allowed=_CONFIG_SECTION_KEYS, context="config")
    doc = dict(component_root_doc)

    experiment_doc = _require_mapping(doc["experiment"], context="experiment") if "experiment" in doc else None
    system_doc = _require_mapping(doc["system"], context="system") if "system" in doc else None
    model_doc = _require_mapping(doc["model"], context="model") if "model" in doc else None
    training_doc = _require_mapping(doc["training"], context="training") if "training" in doc else None
    environment_doc = _require_mapping(doc["environment"], context="environment") if "environment" in doc else None
    rewards_doc = _require_mapping(doc["rewards"], context="rewards") if "rewards" in doc else None
    league_doc = _require_mapping(doc["league"], context="league") if "league" in doc else None
    evaluation_doc = _require_mapping(doc["evaluation"], context="evaluation") if "evaluation" in doc else None
    reproducibility_doc = (
        _require_mapping(doc["reproducibility"], context="reproducibility") if "reproducibility" in doc else None
    )
    curriculum_doc = _require_mapping(doc["curriculum"], context="curriculum") if "curriculum" in doc else None

    experiment = _parse_experiment_config(experiment_doc) if experiment_doc is not None else None
    system = _parse_system_config(system_doc) if system_doc is not None else None
    model = _parse_model_config(model_doc) if model_doc is not None else None
    training = _parse_training_config(training_doc) if training_doc is not None else None
    environment = _parse_environment_config(environment_doc) if environment_doc is not None else None
    rewards = _parse_rewards_config(rewards_doc) if rewards_doc is not None else None
    curriculum = _parse_curriculum_config(curriculum_doc)
    league = _parse_league_config(league_doc) if league_doc is not None else None
    evaluation = _parse_evaluation_config(evaluation_doc) if evaluation_doc is not None else None
    reproducibility = _parse_reproducibility_config(reproducibility_doc) if reproducibility_doc is not None else None

    component_docs = {
        key: _require_mapping(value, context=key) for key, value in doc.items() if key in _CONFIG_SECTION_KEYS
    }
    seed_sets = (
        _parse_seed_sets_override(root=root, seed_sets_doc=seed_sets_override)
        if seed_sets_override is not None
        else _resolve_seed_sets(root=root, league=league, evaluation=evaluation, reproducibility=reproducibility)
    )

    return StackConfig(
        root=root,
        schema_version=schema_version,
        description=_require_text(description, field_name="description"),
        lock_intent={} if lock_intent is None else dict(lock_intent),
        components={},
        seed_sets=seed_sets,
        component_docs=component_docs,
        config=LockedConfig(
            experiment=experiment,
            system=system,
            model=model,
            training=training,
            environment=environment,
            rewards=rewards,
            curriculum=curriculum,
            league=league,
            evaluation=evaluation,
            reproducibility=reproducibility,
        ),
    )


def _load_canonical_stack_config(stack_file: Path) -> StackConfig:
    root = _resolve_repo_root(stack_file)
    payload = _load_json(stack_file)
    _reject_unknown_keys(payload, allowed=_CANONICAL_CONFIG_KEYS, context=str(stack_file))
    config_doc = _require_mapping(payload["config"], context="config")
    seed_sets_doc = _require_mapping(payload["seed_sets"], context="seed_sets") if "seed_sets" in payload else None
    schema_version = (
        _require_int(payload["schema_version"], field_name="schema_version", minimum=1)
        if "schema_version" in payload
        else None
    )
    description = _require_text(payload.get("description", stack_file.stem), field_name="description")
    return _build_stack_config_from_component_doc(
        root=root,
        component_root_doc=config_doc,
        description=description,
        schema_version=schema_version,
        seed_sets_override=seed_sets_doc,
        lock_intent={"canonical_config_payload": payload},
    )


def load_stack_config(stack_path: Path | str) -> StackConfig:
    stack_file = Path(stack_path).resolve()
    if stack_file.suffix.lower() == ".json":
        return _load_canonical_stack_config(stack_file)

    root = _resolve_repo_root(stack_file)
    doc = _load_preset_document(stack_file)
    schema_version = (
        _require_int(doc["schema_version"], field_name="schema_version", minimum=1) if "schema_version" in doc else None
    )
    description = _require_text(doc.get("description", stack_file.stem), field_name="description")
    component_doc = {key: value for key, value in doc.items() if key in _CONFIG_SECTION_KEYS}
    return _build_stack_config_from_component_doc(
        root=root,
        component_root_doc=component_doc,
        description=description,
        schema_version=schema_version,
    )
