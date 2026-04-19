"""Study-only config loading for metagame and sensitivity reporting."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

from .models import (
    MetagameAlphaRankConfig,
    MetagameConfig,
    MetagameNashConfig,
    SensitivityCaseConfig,
    SensitivityConfig,
    SensitivityReportConfig,
    StudyConfig,
)

_TOP_LEVEL_KEYS = frozenset({"schema_version", "description", "metagame", "sensitivity"})


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _resolve_repo_root(config_file: Path) -> Path:
    for candidate in config_file.resolve().parents:
        if (candidate / "configs").is_dir() and (candidate / "python").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve repo root for study config path: {config_file}")


def _reject_unknown_keys(body: Mapping[str, Any], *, allowed: Collection[str], context: str) -> None:
    unknown = sorted(key for key in body if key not in allowed)
    if unknown:
        raise ValueError(f"{context} has unsupported keys: {', '.join(unknown)}")


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def _require_str_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_require_text(item, field_name=f"{field_name}[]") for item in value)


def _parse_metagame_config(body: dict[str, Any]) -> MetagameConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "payoff_uncertainty_method",
            "sampling_M",
            "optional_secondary_uncertainty_method",
            "dirichlet_alpha_wldt",
            "primary_analysis",
            "secondary_analysis",
            "nash",
            "alpharank",
        },
        context="metagame",
    )
    nash = _require_mapping(body["nash"], context="metagame.nash")
    alpharank = _require_mapping(body["alpharank"], context="metagame.alpharank")
    _reject_unknown_keys(
        nash,
        allowed={"impl", "backend", "threads", "value_tolerance", "tie_break"},
        context="metagame.nash",
    )
    _reject_unknown_keys(
        alpharank,
        allowed={"impl", "m", "alpha", "local_selection", "use_inf_alpha", "inf_alpha_eps"},
        context="metagame.alpharank",
    )
    return MetagameConfig(
        payoff_uncertainty_method=_require_text(
            body["payoff_uncertainty_method"],
            field_name="metagame.payoff_uncertainty_method",
        ),
        sampling_m=_require_int(body["sampling_M"], field_name="metagame.sampling_M", minimum=1),
        optional_secondary_uncertainty_method=_require_text(
            body["optional_secondary_uncertainty_method"],
            field_name="metagame.optional_secondary_uncertainty_method",
        ),
        dirichlet_alpha_wldt=_require_float(
            body["dirichlet_alpha_wldt"],
            field_name="metagame.dirichlet_alpha_wldt",
        ),
        primary_analysis=_require_text(body["primary_analysis"], field_name="metagame.primary_analysis"),
        secondary_analysis=_require_text(body["secondary_analysis"], field_name="metagame.secondary_analysis"),
        nash=MetagameNashConfig(
            impl=_require_text(nash["impl"], field_name="metagame.nash.impl"),
            backend=_require_text(nash["backend"], field_name="metagame.nash.backend"),
            threads=_require_int(nash["threads"], field_name="metagame.nash.threads", minimum=1),
            value_tolerance=_require_float(
                nash["value_tolerance"],
                field_name="metagame.nash.value_tolerance",
            ),
            tie_break=_require_text(nash["tie_break"], field_name="metagame.nash.tie_break"),
        ),
        alpharank=MetagameAlphaRankConfig(
            impl=_require_text(alpharank["impl"], field_name="metagame.alpharank.impl"),
            m=_require_int(alpharank["m"], field_name="metagame.alpharank.m", minimum=1),
            alpha=_require_float(alpharank["alpha"], field_name="metagame.alpharank.alpha"),
            local_selection=_require_bool(
                alpharank["local_selection"],
                field_name="metagame.alpharank.local_selection",
            ),
            use_inf_alpha=_require_bool(alpharank["use_inf_alpha"], field_name="metagame.alpharank.use_inf_alpha"),
            inf_alpha_eps=_require_float(
                alpharank["inf_alpha_eps"],
                field_name="metagame.alpharank.inf_alpha_eps",
            ),
        ),
    )


def _parse_sensitivity_config(body: dict[str, Any]) -> SensitivityConfig:
    report = _require_mapping(body.get("report", {}), context="sensitivity.report")
    _reject_unknown_keys(report, allowed={"required_outputs"}, context="sensitivity.report")
    cases: dict[str, SensitivityCaseConfig] = {}
    for case_id, raw_case in body.items():
        if case_id == "report":
            continue
        case = _require_mapping(raw_case, context=f"sensitivity.{case_id}")
        _reject_unknown_keys(
            case,
            allowed={"draw_score", "truncation_score", "truncation_handling", "description"},
            context=f"sensitivity.{case_id}",
        )
        cases[case_id] = SensitivityCaseConfig(
            description=_require_text(case["description"], field_name=f"sensitivity.{case_id}.description"),
            draw_score=_require_float(case["draw_score"], field_name=f"sensitivity.{case_id}.draw_score"),
            truncation_score=(
                None
                if "truncation_score" not in case
                else _require_float(case["truncation_score"], field_name=f"sensitivity.{case_id}.truncation_score")
            ),
            truncation_handling=(
                None
                if "truncation_handling" not in case
                else _require_text(
                    case["truncation_handling"],
                    field_name=f"sensitivity.{case_id}.truncation_handling",
                )
            ),
        )
    return SensitivityConfig(
        cases=cases,
        report=SensitivityReportConfig(
            required_outputs=_require_str_list(
                report.get("required_outputs", []),
                field_name="sensitivity.report.required_outputs",
            )
        ),
    )


def load_study_config(path: Path | str) -> StudyConfig:
    config_file = Path(path).resolve()
    root = _resolve_repo_root(config_file)
    doc = _load_yaml(config_file)
    _reject_unknown_keys(doc, allowed=_TOP_LEVEL_KEYS, context=str(config_file))
    metagame = _parse_metagame_config(_require_mapping(doc["metagame"], context="metagame"))
    sensitivity = _parse_sensitivity_config(_require_mapping(doc["sensitivity"], context="sensitivity"))
    schema_version = (
        None
        if "schema_version" not in doc
        else _require_int(
            doc["schema_version"],
            field_name="schema_version",
            minimum=1,
        )
    )
    description = _require_text(doc.get("description", config_file.stem), field_name="description")
    return StudyConfig(
        root=root,
        schema_version=schema_version,
        description=description,
        metagame=metagame,
        sensitivity=sensitivity,
    )
