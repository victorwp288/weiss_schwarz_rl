"""Compatibility imports for the simulator reward contract."""

from weiss_rl.envs.simulator_reward_contract import (
    SUPPORTED_SIMULATOR_REWARD_OBJECTIVES,
    SimulatorRewardPayload,
    reward_payload_json_from_stack,
    simulator_reward_payload_from_rewards,
)

__all__ = [
    "SUPPORTED_SIMULATOR_REWARD_OBJECTIVES",
    "SimulatorRewardPayload",
    "reward_payload_json_from_stack",
    "simulator_reward_payload_from_rewards",
]
