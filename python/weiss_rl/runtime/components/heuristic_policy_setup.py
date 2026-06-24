"""Build runtime heuristic policies used by teacher and opponent lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.eval.heuristic_public.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    heuristic_public_policy_ids,
    heuristic_public_profile_name_for_policy_id,
)
from weiss_rl.runtime.components.teacher_settings import RuntimeTeacherSettings


@dataclass(frozen=True)
class RuntimeHeuristicPolicySetup:
    teacher_policy: HeuristicPublicPolicy | None
    teacher_policy_by_profile: dict[str, HeuristicPublicPolicy]
    teacher_action_catalog: ActionCatalog | None
    teacher_family_index: dict[str, int]
    teacher_attack_type_index: dict[str, int]
    opponent_heuristic_policies: dict[str, HeuristicPublicPolicy]


def build_runtime_heuristic_policy_setup(
    *,
    spec_bundle: dict[str, Any] | None,
    action_catalog: ActionCatalog | None,
    teacher_settings: RuntimeTeacherSettings,
    actor_policy_backend: str,
    league_config: Any | None,
    diverse_opponent_actor_count: int,
    actor_count: int,
) -> RuntimeHeuristicPolicySetup:
    teacher_policy = _build_teacher_policy(
        spec_bundle=spec_bundle,
        action_catalog=action_catalog,
        teacher_settings=teacher_settings,
        actor_policy_backend=actor_policy_backend,
    )
    opponent_heuristic_policies = _build_opponent_heuristic_policies(
        spec_bundle=spec_bundle,
        league_config=league_config,
        diverse_opponent_actor_count=diverse_opponent_actor_count,
        actor_count=actor_count,
    )
    return RuntimeHeuristicPolicySetup(
        teacher_policy=teacher_policy.teacher_policy,
        teacher_policy_by_profile=teacher_policy.teacher_policy_by_profile,
        teacher_action_catalog=teacher_policy.teacher_action_catalog,
        teacher_family_index=teacher_policy.teacher_family_index,
        teacher_attack_type_index=teacher_policy.teacher_attack_type_index,
        opponent_heuristic_policies=opponent_heuristic_policies,
    )


@dataclass(frozen=True)
class _RuntimeTeacherPolicySetup:
    teacher_policy: HeuristicPublicPolicy | None
    teacher_policy_by_profile: dict[str, HeuristicPublicPolicy]
    teacher_action_catalog: ActionCatalog | None
    teacher_family_index: dict[str, int]
    teacher_attack_type_index: dict[str, int]


def _build_teacher_policy(
    *,
    spec_bundle: dict[str, Any] | None,
    action_catalog: ActionCatalog | None,
    teacher_settings: RuntimeTeacherSettings,
    actor_policy_backend: str,
) -> _RuntimeTeacherPolicySetup:
    teacher_policy: HeuristicPublicPolicy | None = None
    teacher_policy_by_profile: dict[str, HeuristicPublicPolicy] = {}
    teacher_action_catalog: ActionCatalog | None = None
    teacher_family_index: dict[str, int] = {}
    teacher_attack_type_index: dict[str, int] = {}

    if teacher_settings.teacher_guidance_enabled:
        if spec_bundle is None:
            raise RuntimeError("structured_aux.enabled requires the runtime spec bundle")
        try:
            teacher_policy = HeuristicPublicPolicy.from_spec_bundle(spec_bundle)
            teacher_policy_by_profile["base"] = teacher_policy
            for profile_name in teacher_settings.teacher_label_profiles:
                normalized_profile = str(profile_name).strip().lower()
                if not normalized_profile or normalized_profile == "base":
                    continue
                teacher_policy_by_profile[normalized_profile] = HeuristicPublicPolicy.from_spec_bundle(
                    spec_bundle,
                    scoring_profile=normalized_profile,
                )
            teacher_action_catalog = action_catalog or ActionCatalog.from_spec_bundle(spec_bundle)
        except Exception as exc:
            raise RuntimeError(
                "Structured teacher guidance requires a heuristic-compatible simulator contract"
            ) from exc
        teacher_family_index = {family.name: index for index, family in enumerate(teacher_action_catalog.families)}
        teacher_attack_type_index = {name: index for index, name in enumerate(teacher_action_catalog.attack_type_names)}

    if actor_policy_backend == "heuristic_public" and teacher_policy is None:
        if spec_bundle is None:
            raise RuntimeError("training.actor_policy_backend=heuristic_public requires the runtime spec bundle")
        teacher_policy = HeuristicPublicPolicy.from_spec_bundle(spec_bundle)

    return _RuntimeTeacherPolicySetup(
        teacher_policy=teacher_policy,
        teacher_policy_by_profile=teacher_policy_by_profile,
        teacher_action_catalog=teacher_action_catalog,
        teacher_family_index=teacher_family_index,
        teacher_attack_type_index=teacher_attack_type_index,
    )


def _build_opponent_heuristic_policies(
    *,
    spec_bundle: dict[str, Any] | None,
    league_config: Any | None,
    diverse_opponent_actor_count: int,
    actor_count: int,
) -> dict[str, HeuristicPublicPolicy]:
    opponent_heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
    heuristic_public_mix_fraction, heuristic_public_variant_mix_fraction = _heuristic_public_mix_fractions(
        league_config
    )
    base_heuristic_required = bool(
        heuristic_public_mix_fraction > 0.0
        or (int(diverse_opponent_actor_count) > 0 and int(diverse_opponent_actor_count) < int(actor_count))
    )
    if base_heuristic_required:
        if spec_bundle is None:
            raise RuntimeError("heuristic-public opponent lanes require the runtime spec bundle")
        try:
            opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = HeuristicPublicPolicy.from_spec_bundle(
                spec_bundle
            )
        except Exception as exc:
            raise RuntimeError(
                "Training-time B2 HeuristicPublic requires a heuristic-compatible simulator contract"
            ) from exc

    if heuristic_public_variant_mix_fraction > 0.0:
        if spec_bundle is None:
            raise RuntimeError(
                "league.sampling.heuristic_public_variant_mix_fraction > 0 requires the runtime spec bundle"
            )
        try:
            for policy_id in heuristic_public_policy_ids(include_base=False):
                profile_name = heuristic_public_profile_name_for_policy_id(policy_id)
                if profile_name is None:
                    continue
                opponent_heuristic_policies[policy_id] = HeuristicPublicPolicy.from_spec_bundle(
                    spec_bundle,
                    scoring_profile=profile_name,
                )
        except Exception as exc:
            raise RuntimeError(
                "Training-time heuristic-public variant baselines require a heuristic-compatible simulator contract"
            ) from exc

    return opponent_heuristic_policies


def _heuristic_public_mix_fractions(league_config: Any | None) -> tuple[float, float]:
    if league_config is None:
        return 0.0, 0.0
    sampling_cfg = getattr(league_config, "sampling", league_config)
    heuristic_public_mix_fraction = float(getattr(sampling_cfg, "heuristic_public_mix_fraction", 0.0))
    heuristic_public_variant_mix_fraction = max(
        float(getattr(sampling_cfg, "heuristic_public_variant_mix_fraction", 0.0)),
        float(
            getattr(
                sampling_cfg,
                "heuristic_public_variant_final_mix_fraction",
                getattr(sampling_cfg, "heuristic_public_variant_mix_fraction", 0.0),
            )
        ),
    )
    return heuristic_public_mix_fraction, heuristic_public_variant_mix_fraction


__all__ = ["RuntimeHeuristicPolicySetup", "build_runtime_heuristic_policy_setup"]
