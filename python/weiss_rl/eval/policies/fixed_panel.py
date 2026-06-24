"""Fixed public policy IDs, deck bindings, and heuristic profiles."""

from __future__ import annotations

from dataclasses import dataclass

RANDOM_LEGAL_POLICY_ID = "B0 RandomLegal"
NO_LEAGUE_POLICY_ID = "B1 NoLeague baseline"
LEGACY_NO_LEAGUE_POLICY_ID = "b1_noleague_baseline"
HEURISTIC_PUBLIC_POLICY_ID = "B2 HeuristicPublic"
HEURISTIC_PUBLIC_AGGRO_POLICY_ID = "B3 HeuristicPublicAggro"
HEURISTIC_PUBLIC_CONTROL_POLICY_ID = "B4 HeuristicPublicControl"

MAIN_DECK_ID = "preset:main_deck_5hy_yotsuba_v1"
STARTER_DECK_ID = "preset:starter_deck_ws02_v1"
AGGRO_DECK_ID = "preset:aggro_deck_5hy_nino_v1"
CONTROL_DECK_ID = "preset:control_deck_jj_s66_v1"

_HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID = {
    HEURISTIC_PUBLIC_POLICY_ID: "base",
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID: "aggressive",
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID: "control",
}

_EVAL_DECK_BY_POLICY_ID = {
    RANDOM_LEGAL_POLICY_ID: MAIN_DECK_ID,
    NO_LEAGUE_POLICY_ID: MAIN_DECK_ID,
    LEGACY_NO_LEAGUE_POLICY_ID: MAIN_DECK_ID,
    HEURISTIC_PUBLIC_POLICY_ID: MAIN_DECK_ID,
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID: AGGRO_DECK_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID: CONTROL_DECK_ID,
}


@dataclass(frozen=True, slots=True)
class FixedPanelPolicyRole:
    policy_id: str
    role: str
    deck_id: str
    policy_source: str
    evidence_question: str

    def as_payload(self) -> dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "role": self.role,
            "deck_id": self.deck_id,
            "policy_source": self.policy_source,
            "evidence_question": self.evidence_question,
        }


FIXED_POLICY_PANEL_ROLES: tuple[FixedPanelPolicyRole, ...] = (
    FixedPanelPolicyRole(
        policy_id=RANDOM_LEGAL_POLICY_ID,
        role="B0 random legal baseline",
        deck_id=MAIN_DECK_ID,
        policy_source="built-in random legal policy",
        evidence_question="Does the trained policy beat legal random play on the main deck?",
    ),
    FixedPanelPolicyRole(
        policy_id=NO_LEAGUE_POLICY_ID,
        role="B1 retained no-league baseline",
        deck_id=MAIN_DECK_ID,
        policy_source="selected retained B1 checkpoint",
        evidence_question="Does the main policy improve over the explicit B1 anchor?",
    ),
    FixedPanelPolicyRole(
        policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        role="B2 public heuristic on the main deck",
        deck_id=MAIN_DECK_ID,
        policy_source="public-state heuristic profile: base",
        evidence_question="Does the policy transfer to a stronger public heuristic opponent?",
    ),
    FixedPanelPolicyRole(
        policy_id=HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        role="B3 public heuristic on the aggro deck",
        deck_id=AGGRO_DECK_ID,
        policy_source="public-state heuristic profile: aggressive",
        evidence_question="Does the policy survive a different deck pressure profile?",
    ),
    FixedPanelPolicyRole(
        policy_id=HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
        role="B4 public heuristic on the control deck",
        deck_id=CONTROL_DECK_ID,
        policy_source="public-state heuristic profile: control",
        evidence_question="Does the policy remain robust against a slower control profile?",
    ),
)


def fixed_policy_panel_role_payload() -> list[dict[str, str]]:
    return [role.as_payload() for role in FIXED_POLICY_PANEL_ROLES]


def heuristic_public_profile_name_for_policy_id(policy_id: str) -> str | None:
    """Return the heuristic scoring profile bound to a B2-B4 policy ID."""

    return _HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID.get(str(policy_id))


def heuristic_public_policy_ids(*, include_base: bool = True) -> tuple[str, ...]:
    """Return fixed heuristic-public anchor IDs in panel order."""

    policy_ids = tuple(_HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID)
    if include_base:
        return policy_ids
    return tuple(policy_id for policy_id in policy_ids if policy_id != HEURISTIC_PUBLIC_POLICY_ID)


def deck_id_for_policy_id(policy_id: str) -> str:
    """Return the eval deck used for a fixed policy ID or the default main deck."""

    return _EVAL_DECK_BY_POLICY_ID.get(str(policy_id), MAIN_DECK_ID)


__all__ = [
    "AGGRO_DECK_ID",
    "CONTROL_DECK_ID",
    "FIXED_POLICY_PANEL_ROLES",
    "FixedPanelPolicyRole",
    "HEURISTIC_PUBLIC_AGGRO_POLICY_ID",
    "HEURISTIC_PUBLIC_CONTROL_POLICY_ID",
    "HEURISTIC_PUBLIC_POLICY_ID",
    "LEGACY_NO_LEAGUE_POLICY_ID",
    "MAIN_DECK_ID",
    "NO_LEAGUE_POLICY_ID",
    "RANDOM_LEGAL_POLICY_ID",
    "STARTER_DECK_ID",
    "deck_id_for_policy_id",
    "fixed_policy_panel_role_payload",
    "heuristic_public_policy_ids",
    "heuristic_public_profile_name_for_policy_id",
]
