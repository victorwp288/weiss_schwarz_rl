"""QueueRuntime opponent/PFSP adapter methods.

The sampling and bookkeeping algorithms live in :mod:`weiss_rl.runtime.components.opponents`.
This mixin only adapts QueueRuntime instance state into those pure helpers while
preserving the private method names historically available on ``QueueRuntime``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID, heuristic_public_policy_ids
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.league.opponent_pool import (
    OpponentPoolSampler,
    compose_runtime_opponent_pool,
    select_runtime_opponent_snapshots,
)
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.runtime.components.opponents import (
    active_actor_heuristic_fraction,
    active_assigned_opponent_policy_ids,
    active_heuristic_public_mix_fraction,
    active_heuristic_public_variant_mix_fraction,
    active_mirror_mix_fraction,
    active_noleague_baseline_mix_fraction,
    active_warmup_snapshot_mix_fraction,
    apply_opponent_pool_diversity_floor,
    configured_fixed_opponent_policy_ids,
    configured_hard_negative_focus_policy_ids,
    configured_resident_opponent_policy_ids,
    configured_row_deficit_policy_weights,
    filter_timeout_heavy_opponents,
    fixed_opponent_policy_is_active,
    fixed_opponent_policy_slots,
    promotion_gated_recent_reservoir_size,
    sample_runtime_opponent_policy_ids,
    sample_warmup_snapshot_policy_ids,
    select_hard_negative_ids,
)
from weiss_rl.runtime.components.policy_ids import FIXED_OPPONENT_EXCLUSIONS, MIRROR_OPPONENT_POLICY_ID

_HEURISTIC_PUBLIC_VARIANT_POLICY_IDS = heuristic_public_policy_ids(include_base=False)
_PFSP_TIMEOUT_FILTER_MIN_SAMPLES = 32
_PROMOTION_GATED_RECENT_RESERVOIR_MIN_SIZE = 2
_PFSP_DIVERSITY_FLOOR_SIZE = 2


class QueueRuntimeOpponentMixin:
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _active_assigned_opponent_policy_ids(self) -> tuple[str, ...]:
        return active_assigned_opponent_policy_ids(
            actors=tuple(getattr(self, "_actors", ())),
            mirror_policy_id=MIRROR_OPPONENT_POLICY_ID,
        )

    def _configured_fixed_opponent_policy_ids(self) -> tuple[str, ...]:
        return configured_fixed_opponent_policy_ids(
            heuristic_reserved_envs_per_actor=int(getattr(self, "_heuristic_public_reserved_envs_per_actor", 0)),
            noleague_baseline_reserved_envs_per_actor=int(
                getattr(self, "_noleague_baseline_reserved_envs_per_actor", 0)
            ),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            noleague_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
            heuristic_policy_ids=tuple(getattr(self, "_opponent_heuristic_policies", {}).keys()),
        )

    def _configured_resident_opponent_policy_ids(self) -> tuple[str, ...]:
        return configured_resident_opponent_policy_ids(
            fixed_policy_ids=self._configured_fixed_opponent_policy_ids(),
            heuristic_variant_mix_fraction=self._active_heuristic_public_variant_mix_fraction(),
            noleague_mix_fraction=self._active_noleague_baseline_mix_fraction(),
            heuristic_variant_policy_ids=_HEURISTIC_PUBLIC_VARIANT_POLICY_IDS,
            heuristic_policy_ids=tuple(getattr(self, "_opponent_heuristic_policies", {}).keys()),
            noleague_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
        )

    def _active_noleague_baseline_mix_fraction(self) -> float:
        return active_noleague_baseline_mix_fraction(
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
        )

    def _active_heuristic_public_mix_fraction(self) -> float:
        return active_heuristic_public_mix_fraction(
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
        )

    def _active_heuristic_public_variant_mix_fraction(self) -> float:
        return active_heuristic_public_variant_mix_fraction(
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
        )

    def _active_mirror_mix_fraction(self) -> float:
        return active_mirror_mix_fraction(
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
        )

    def _active_warmup_snapshot_mix_fraction(self) -> float:
        return active_warmup_snapshot_mix_fraction(
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
            has_opponent_candidates=bool(getattr(self, "_opponent_candidate_ids", ())),
            has_opponent_models=bool(getattr(self, "_opponent_models", {})),
        )

    def _active_actor_heuristic_fraction(self) -> float:
        initial_fraction = float(getattr(self, "_actor_heuristic_fraction", 1.0))
        return active_actor_heuristic_fraction(
            initial_fraction=initial_fraction,
            final_fraction=float(getattr(self, "_actor_heuristic_final_fraction", initial_fraction)),
            start_updates=int(getattr(self, "_actor_heuristic_start_updates", 0)),
            end_updates=int(getattr(self, "_actor_heuristic_end_updates", -1)),
            reference_update=self._league_reference_update(),
        )

    def _fixed_opponent_policy_slots(self) -> np.ndarray | None:
        return fixed_opponent_policy_slots(
            envs_per_actor=int(self.config.envs_per_actor),
            heuristic_reserved_envs=int(getattr(self, "_heuristic_public_reserved_envs_per_actor", 0)),
            noleague_reserved_envs=int(getattr(self, "_noleague_baseline_reserved_envs_per_actor", 0)),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            noleague_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
        )

    def _fixed_opponent_policy_is_active(self, policy_id: str) -> bool:
        return fixed_opponent_policy_is_active(
            policy_id=policy_id,
            forced_policy_ids=tuple(getattr(self, "_forced_fixed_opponent_policy_ids", ())),
            heuristic_policy_ids=tuple(self._opponent_heuristic_policies.keys()),
            opponent_model_ids=tuple(self._opponent_models.keys()),
            league_config=self._league_config,
            reference_update=self._league_reference_update(),
            noleague_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
        )

    def _promotion_gated_recent_reservoir_size(
        self,
        *,
        base_recent_size: int,
        champion_size: int,
        admitted_champion_ids: Sequence[str],
    ) -> int:
        return promotion_gated_recent_reservoir_size(
            base_recent_size=base_recent_size,
            champion_size=champion_size,
            admitted_champion_ids=admitted_champion_ids,
            min_recent_size=_PROMOTION_GATED_RECENT_RESERVOIR_MIN_SIZE,
        )

    def _filter_timeout_heavy_opponents(self, candidate_ids: Sequence[str]) -> tuple[str, ...]:
        return filter_timeout_heavy_opponents(
            candidate_ids=candidate_ids,
            league_config=self._league_config,
            outcomes=getattr(self, "_outcomes", None),
            min_samples=_PFSP_TIMEOUT_FILTER_MIN_SAMPLES,
        )

    def _apply_opponent_pool_diversity_floor(
        self,
        *,
        candidate_ids: Sequence[str],
        filtered_candidate_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], int]:
        return apply_opponent_pool_diversity_floor(
            candidate_ids=candidate_ids,
            filtered_candidate_ids=filtered_candidate_ids,
            minimum_floor_size=_PFSP_DIVERSITY_FLOOR_SIZE,
        )

    def _select_hard_negative_ids(self, candidate_ids: Sequence[str]) -> tuple[str, ...]:
        return select_hard_negative_ids(
            candidate_ids=candidate_ids,
            league_config=self._league_config,
            outcomes=getattr(self, "_outcomes", None),
            registry_path=getattr(self, "_registry_path", None),
        )

    def _sample_opponent_policy_ids(self, *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        if int(count) <= 0 or not bool(self._league_enabled):
            result = sample_runtime_opponent_policy_ids(
                count=count,
                rng=rng,
                league_enabled=bool(self._league_enabled),
                league_config=getattr(self, "_league_config", None),
                pfsp_ready=False,
                reference_update=0,
                mirror_weight=0.0,
                heuristic_public_weight=0.0,
                heuristic_public_variant_weight=0.0,
                noleague_baseline_weight=0.0,
                warmup_snapshot_weight=0.0,
                opponent_candidate_ids=(),
                opponent_hard_negative_ids=(),
                opponent_champion_ids=(),
                opponent_recent_ids=(),
                opponent_heuristic_policy_ids=(),
                opponent_model_ids=(),
                outcomes=None,
                mirror_policy_id=MIRROR_OPPONENT_POLICY_ID,
                heuristic_public_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
                heuristic_public_variant_policy_ids=_HEURISTIC_PUBLIC_VARIANT_POLICY_IDS,
                noleague_baseline_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
            )
        else:
            result = sample_runtime_opponent_policy_ids(
                count=count,
                rng=rng,
                league_enabled=True,
                league_config=getattr(self, "_league_config", None),
                pfsp_ready=self._pfsp_sampling_ready(),
                reference_update=self._league_reference_update(),
                mirror_weight=self._active_mirror_mix_fraction(),
                heuristic_public_weight=self._active_heuristic_public_mix_fraction(),
                heuristic_public_variant_weight=self._active_heuristic_public_variant_mix_fraction(),
                noleague_baseline_weight=self._active_noleague_baseline_mix_fraction(),
                warmup_snapshot_weight=self._active_warmup_snapshot_mix_fraction(),
                opponent_candidate_ids=getattr(self, "_opponent_candidate_ids", ()),
                opponent_hard_negative_ids=getattr(self, "_opponent_hard_negative_ids", ()),
                opponent_champion_ids=getattr(self, "_opponent_champion_ids", ()),
                opponent_recent_ids=getattr(self, "_opponent_recent_ids", ()),
                opponent_heuristic_policy_ids=tuple(getattr(self, "_opponent_heuristic_policies", {}).keys()),
                opponent_model_ids=tuple(getattr(self, "_opponent_models", {}).keys()),
                outcomes=self._outcomes,
                mirror_policy_id=MIRROR_OPPONENT_POLICY_ID,
                heuristic_public_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
                heuristic_public_variant_policy_ids=_HEURISTIC_PUBLIC_VARIANT_POLICY_IDS,
                noleague_baseline_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
            )
        self._record_opponent_sampling_result(result)
        return result.policy_ids

    def _sample_warmup_snapshot_policy_ids(self, *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        opponent_candidate_ids = getattr(self, "_opponent_candidate_ids", ())
        result = sample_warmup_snapshot_policy_ids(
            count=count,
            rng=rng,
            opponent_candidate_ids=opponent_candidate_ids,
            league_config=getattr(self, "_league_config", None),
            outcomes=None if int(count) <= 0 or not opponent_candidate_ids else self._outcomes,
        )
        self._record_opponent_sampling_result(result)
        return result.policy_ids

    def _record_opponent_sampling_result(self, result) -> None:
        self._pfsp_last_sampled_envs = result.sampled_envs
        self._pfsp_last_mirror_envs = result.mirror_envs
        self._pfsp_last_heuristic_public_envs = result.heuristic_public_envs
        self._pfsp_last_heuristic_public_variant_envs = result.heuristic_public_variant_envs
        self._pfsp_last_noleague_baseline_envs = result.noleague_baseline_envs
        self._pfsp_last_hard_negative_envs = result.hard_negative_envs
        self._pfsp_last_champion_envs = result.champion_envs
        self._pfsp_last_recent_envs = result.recent_envs
        self._pfsp_last_warmup_snapshot_envs = result.warmup_snapshot_envs
        self._pfsp_last_sampled_policy_envs = dict(result.sampled_policy_envs)
        self._pfsp_last_heuristic_public_policy_envs = dict(result.heuristic_public_policy_envs)
        self._pfsp_last_heuristic_public_variant_policy_envs = dict(result.heuristic_public_variant_policy_envs)
        self._pfsp_last_noleague_baseline_policy_envs = dict(result.noleague_baseline_policy_envs)
        self._pfsp_last_champion_policy_envs = dict(result.champion_policy_envs)
        self._pfsp_last_recent_policy_envs = dict(result.recent_policy_envs)
        self._pfsp_last_hard_negative_policy_envs = dict(result.hard_negative_policy_envs)
        self._pfsp_last_warmup_snapshot_policy_envs = dict(result.warmup_snapshot_policy_envs)

    def _pfsp_sampling_ready(self) -> bool:
        if not self._league_enabled or self._league_config is None or self._opponent_sampler is None:
            return False
        if self._league_reference_update() < int(self._league_config.warmup.first_updates):
            return False
        return bool(self._opponent_candidate_ids) and bool(self._opponent_models)

    def refresh_opponent_pool(self) -> None:
        if not self._league_enabled or self._registry_path is None or not self._registry_path.is_file():
            empty_policy_ids: tuple[str, ...] = ()
            self._opponent_sampler = None
            self._opponent_candidate_ids = empty_policy_ids
            self._opponent_models = {}
            self._opponent_model_locks = {}
            self._pfsp_pool_size = 0
            self._pfsp_quarantined_opponents = 0
            self._pfsp_champion_pool_size = 0
            self._pfsp_recent_pool_size = 0
            self._pfsp_hard_negative_pool_size = 0
            self._opponent_champion_ids = empty_policy_ids
            self._opponent_recent_ids = empty_policy_ids
            self._opponent_hard_negative_ids = empty_policy_ids
            self._write_opponent_pool_refresh_record(
                current_update=int(self._league_reference_update()),
                registry_path=getattr(self, "_registry_path", None),
                candidate_ids=empty_policy_ids,
                champion_ids=empty_policy_ids,
                recent_ids=empty_policy_ids,
                hard_negative_ids=empty_policy_ids,
                resident_policy_ids=empty_policy_ids,
                loaded_model_ids=empty_policy_ids,
                stale_demoted=empty_policy_ids,
                quarantined_count=0,
                reason="disabled_or_missing_registry",
            )
            if getattr(self, "_collector_result_queue", None) is not None:
                for control_queue in getattr(self, "_collector_control_queues", ()):
                    control_queue.put({"kind": "refresh_opponent_pool"})
            return
        assert self._league_config is not None
        pool_cfg = getattr(self._league_config, "pool", self._league_config)
        current_update = int(self._league_reference_update())
        registry = SnapshotRegistry.load(self._registry_path)
        stale_demoted: list[str] = []
        max_age_updates = int(getattr(pool_cfg, "champion_max_age_updates", 0))
        if max_age_updates > 0 and current_update > 0:
            stale_demoted = registry.demote_stale_champions(
                current_update=current_update,
                max_age_updates=max_age_updates,
            )
        admitted_champion_ids = tuple(
            registry.latest_champions(
                int(self._league_config.snapshot_pool_champion_size),
                current_update=current_update,
                max_age_updates=max_age_updates,
            )
        )
        recent_size = int(self._league_config.snapshot_pool_recent_size)
        # Promotion gating should keep rejected snapshots out of the steady-state live PFSP pool.
        # However, if no champion has been admitted yet, forcing recent_size=0 collapses training to
        # mirror-only self-play. Keep a small probationary reservoir before the first champion and a
        # small exploration reservoir afterward so the live pool cannot narrow to champions only.
        if bool(self._league_config.promotion_gate_enabled):
            recent_size = self._promotion_gated_recent_reservoir_size(
                base_recent_size=recent_size,
                champion_size=int(self._league_config.snapshot_pool_champion_size),
                admitted_champion_ids=admitted_champion_ids,
            )
        sampler = OpponentPoolSampler(
            registry=registry,
            recent_size=recent_size,
            champion_size=int(self._league_config.snapshot_pool_champion_size),
            power=float(self._league_config.pfsp_power),
            eps_uniform=float(self._league_config.pfsp_epsilon_uniform),
        )
        self._opponent_sampler = sampler
        selection = select_runtime_opponent_snapshots(
            registry,
            recent_size=recent_size,
            champion_ids=admitted_champion_ids,
            excluded_policy_ids=FIXED_OPPONENT_EXCLUSIONS,
        )
        candidate_ids = selection.candidate_ids
        filtered_candidate_ids = self._filter_timeout_heavy_opponents(candidate_ids)
        candidate_ids, quarantined_count = self._apply_opponent_pool_diversity_floor(
            candidate_ids=candidate_ids,
            filtered_candidate_ids=filtered_candidate_ids,
        )
        self._pfsp_quarantined_opponents = quarantined_count
        hard_negative_ids = self._select_hard_negative_ids(candidate_ids)
        sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
        hard_negative_overlaps_champions = bool(getattr(sampling_cfg, "hard_negative_overlaps_champions", False))
        composition = compose_runtime_opponent_pool(
            selection=selection,
            candidate_ids=candidate_ids,
            hard_negative_ids=hard_negative_ids,
            hard_negative_overlaps_champions=hard_negative_overlaps_champions,
        )
        candidate_ids = composition.candidate_ids
        champion_ids = composition.champion_ids
        recent_ids = composition.recent_ids
        hard_negative_ids = composition.hard_negative_ids
        self._opponent_candidate_ids = candidate_ids
        self._pfsp_pool_size = len(candidate_ids)
        self._opponent_champion_ids = champion_ids
        self._opponent_recent_ids = recent_ids
        self._opponent_hard_negative_ids = hard_negative_ids
        self._pfsp_champion_pool_size = len(champion_ids)
        self._pfsp_recent_pool_size = len(recent_ids)
        self._pfsp_hard_negative_pool_size = len(hard_negative_ids)
        models: dict[str, Any] = {}
        snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
        resident_policy_ids = tuple(
            dict.fromkeys(
                [
                    *candidate_ids,
                    *self._active_assigned_opponent_policy_ids(),
                    *self._configured_resident_opponent_policy_ids(),
                ]
            )
        )
        for policy_id in resident_policy_ids:
            snapshot = snapshots_by_id.get(policy_id)
            if snapshot is None:
                continue
            models[policy_id] = self._load_snapshot_model(snapshot.path)
        self._opponent_models = models
        self._opponent_model_locks = {policy_id: threading.Lock() for policy_id in models}
        if stale_demoted:
            registry.save(self._registry_path)
        self._write_opponent_pool_refresh_record(
            current_update=current_update,
            registry_path=self._registry_path,
            candidate_ids=candidate_ids,
            champion_ids=champion_ids,
            recent_ids=recent_ids,
            hard_negative_ids=hard_negative_ids,
            resident_policy_ids=resident_policy_ids,
            loaded_model_ids=tuple(models.keys()),
            stale_demoted=tuple(stale_demoted),
            quarantined_count=quarantined_count,
            reason="refreshed",
        )
        if getattr(self, "_collector_result_queue", None) is not None:
            for control_queue in getattr(self, "_collector_control_queues", ()):
                control_queue.put({"kind": "refresh_opponent_pool"})

    def _write_opponent_pool_refresh_record(
        self,
        *,
        current_update: int,
        registry_path: Path | None,
        candidate_ids: Sequence[str],
        champion_ids: Sequence[str],
        recent_ids: Sequence[str],
        hard_negative_ids: Sequence[str],
        resident_policy_ids: Sequence[str],
        loaded_model_ids: Sequence[str],
        stale_demoted: Sequence[str],
        quarantined_count: int,
        reason: str,
    ) -> None:
        run_dir = getattr(self, "_run_dir", None)
        if run_dir is None:
            return
        logs_dir = Path(run_dir) / "training" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "opponent_pool_refresh_v1",
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "process_id": int(os.getpid()),
            "reason": str(reason),
            "update": int(current_update),
            "registry_path": self._pool_log_path_text(registry_path),
            "candidate_ids": list(candidate_ids),
            "champion_ids": list(champion_ids),
            "recent_ids": list(recent_ids),
            "hard_negative_ids": list(hard_negative_ids),
            "hard_negative_focus_policy_ids": list(
                configured_hard_negative_focus_policy_ids(league_config=getattr(self, "_league_config", None))
            ),
            "hard_negative_focus_weight_multiplier": float(
                getattr(
                    getattr(getattr(self, "_league_config", None), "sampling", getattr(self, "_league_config", None)),
                    "hard_negative_focus_weight_multiplier",
                    1.0,
                )
            ),
            "row_deficit_policy_weights": [
                [str(policy_id), float(weight)]
                for policy_id, weight in configured_row_deficit_policy_weights(
                    league_config=getattr(self, "_league_config", None)
                )
            ],
            "hard_negative_overlaps_champions": bool(
                getattr(
                    getattr(getattr(self, "_league_config", None), "sampling", getattr(self, "_league_config", None)),
                    "hard_negative_overlaps_champions",
                    False,
                )
            ),
            "resident_policy_ids": list(resident_policy_ids),
            "loaded_model_ids": list(loaded_model_ids),
            "stale_demoted": list(stale_demoted),
            "quarantined_count": int(quarantined_count),
            "pool_size": int(len(candidate_ids)),
            "champion_pool_size": int(len(champion_ids)),
            "recent_pool_size": int(len(recent_ids)),
            "hard_negative_pool_size": int(len(hard_negative_ids)),
        }
        with (logs_dir / "opponent_pool.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _pool_log_path_text(self, path: Path | None) -> str | None:
        if path is None:
            return None
        candidate = Path(path)
        run_dir = getattr(self, "_run_dir", None)
        if run_dir is not None:
            try:
                return candidate.resolve().relative_to(Path(run_dir).resolve()).as_posix()
            except ValueError:
                pass
        return candidate.as_posix()
