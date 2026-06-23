from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.runtime import QueueRuntime


class OpponentSamplingOutcomes:
    _win_rates = {
        "hard_a": 0.2,
        "hard_b": 0.4,
        "champ_a": 0.5,
        "recent_a": 0.6,
        "recent_b": 0.1,
        "warm_a": 0.2,
        "warm_b": 0.8,
    }

    def win_rate(self, policy_id: str) -> float:
        return float(self._win_rates.get(str(policy_id), 0.5))


def make_sampling_runtime(
    *,
    league_config: SimpleNamespace,
    opponent_candidate_ids: tuple[str, ...] = (),
    opponent_hard_negative_ids: tuple[str, ...] = (),
    opponent_champion_ids: tuple[str, ...] = (),
    opponent_recent_ids: tuple[str, ...] = (),
    opponent_heuristic_policies: dict[str, object] | None = None,
    opponent_models: dict[str, object] | None = None,
    pfsp_ready: bool = False,
    reference_update: int = 0,
) -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._league_config = league_config
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_candidate_ids = opponent_candidate_ids
    runtime_any._opponent_hard_negative_ids = opponent_hard_negative_ids
    runtime_any._opponent_champion_ids = opponent_champion_ids
    runtime_any._opponent_recent_ids = opponent_recent_ids
    runtime_any._opponent_heuristic_policies = dict(opponent_heuristic_policies or {})
    runtime_any._opponent_models = dict(opponent_models or {})
    runtime_any._pfsp_sampling_ready = lambda: pfsp_ready
    runtime_any._league_reference_update = lambda: reference_update
    reset_sampling_counters(runtime)
    return runtime


def reset_sampling_counters(runtime: QueueRuntime) -> None:
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
