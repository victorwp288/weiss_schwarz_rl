"""Reader-facing reward flow from simulator payload to retained evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardFlowStage:
    name: str
    purpose: str
    owner_modules: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "owner_modules": list(self.owner_modules),
            "evidence": list(self.evidence),
        }


REWARD_FLOW_STAGES: tuple[RewardFlowStage, ...] = (
    RewardFlowStage(
        name="simulator_reward_payload",
        purpose="Translate the stack reward config into the simulator-facing reward JSON.",
        owner_modules=(
            "weiss_rl.envs.env_config",
            "weiss_rl.envs.simulator_reward_contract",
            "weiss_rl.envs.reward_payload",
        ),
        evidence=("rewards config", "reward_json", "terminal/shaping component scales"),
    ),
    RewardFlowStage(
        name="learner_perspective_rows",
        purpose="Keep rewards signed from the focal learner perspective on trainable rows.",
        owner_modules=(
            "weiss_rl.envs.learner_turn_env",
            "weiss_rl.runtime.components.actors.actor_state",
            "weiss_rl.runtime.components.process",
        ),
        evidence=("acting seat", "focal seat", "train mask", "terminated/truncated flags"),
    ),
    RewardFlowStage(
        name="collector_reward_shaping",
        purpose="Apply narrow local action-quality penalties and record shaping counters.",
        owner_modules=(
            "weiss_rl.runtime.components.rewards.reward_shaping",
            "weiss_rl.runtime.components.rewards.reward_shaping_pass",
            "weiss_rl.runtime.components.rewards.reward_shaping_mulligan",
            "weiss_rl.runtime.components.rewards.reward_shaping_counters",
        ),
        evidence=("collector shaping plan", "legal ids or mask", "penalty counters"),
    ),
    RewardFlowStage(
        name="terminal_backfill",
        purpose="Move terminal outcome credit onto eligible learner-batch rows when configured.",
        owner_modules=("weiss_rl.runtime.components.batching.reward_backfill",),
        evidence=("terminal outcome backfill settings", "trace backfill settings", "discount masks"),
    ),
    RewardFlowStage(
        name="reward_component_probe",
        purpose="Check that simulator component sums match rewards at the expected scale.",
        owner_modules=("weiss_rl.diagnostics.probes.reward_component_probe_entrypoint",),
        evidence=("component order", "component sum error", "terminal/truncation fractions"),
    ),
    RewardFlowStage(
        name="evaluation_outcome",
        purpose="Keep final evaluation based on game outcomes instead of training reward shaping.",
        owner_modules=(
            "weiss_rl.eval.final.run",
            "weiss_rl.eval.analysis.payoff_folding",
            "weiss_rl.eval.analysis.uncertainty",
        ),
        evidence=("winner seat", "paired seeds", "folded payoff matrix", "uncertainty summary"),
    ),
)


def reward_flow_stage_payload() -> list[dict[str, object]]:
    return [stage.as_payload() for stage in REWARD_FLOW_STAGES]


__all__ = [
    "REWARD_FLOW_STAGES",
    "RewardFlowStage",
    "reward_flow_stage_payload",
]
