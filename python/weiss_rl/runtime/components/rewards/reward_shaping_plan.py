"""Reward-shaping rule order for runtime collector batches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardShapingRulePlan:
    rule_id: str
    title: str
    purpose: str
    requires_legal_ids: bool
    supports_legal_mask: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "purpose": self.purpose,
            "requires_legal_ids": self.requires_legal_ids,
            "supports_legal_mask": self.supports_legal_mask,
        }


COLLECTOR_REWARD_SHAPING_PLAN = (
    RewardShapingRulePlan(
        rule_id="pass_with_nonpass_penalty",
        title="Pass with productive alternative",
        purpose="Discourage pass when a productive non-pass action is legal.",
        requires_legal_ids=False,
        supports_legal_mask=True,
    ),
    RewardShapingRulePlan(
        rule_id="mulligan_select_with_confirm_penalty",
        title="Mulligan select when confirm is legal",
        purpose="Discourage selecting more mulligan cards once confirm is available.",
        requires_legal_ids=True,
        supports_legal_mask=False,
    ),
)


def collector_reward_shaping_plan_payload() -> list[dict[str, object]]:
    return [rule.as_payload() for rule in COLLECTOR_REWARD_SHAPING_PLAN]


__all__ = [
    "COLLECTOR_REWARD_SHAPING_PLAN",
    "RewardShapingRulePlan",
    "collector_reward_shaping_plan_payload",
]
