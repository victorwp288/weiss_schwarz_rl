from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv, LegalMode

_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


class FakeMaskOut:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, 4), dtype=np.float32)
        self.masks = np.zeros((num_envs, 52), dtype=np.uint8)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=bool)
        self.truncated = np.zeros((num_envs,), dtype=bool)
        self.actor = np.zeros((num_envs,), dtype=np.int32)
        self.decision_kind = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int64)
        self.engine_status = np.zeros((num_envs,), dtype=np.uint8)
        self.spec_hash = np.zeros((num_envs,), dtype=np.uint64)


class FakeNoMaskOut:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, 4), dtype=np.float32)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=bool)
        self.truncated = np.zeros((num_envs,), dtype=bool)
        self.actor = np.zeros((num_envs,), dtype=np.int32)
        self.decision_kind = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int64)
        self.engine_status = np.zeros((num_envs,), dtype=np.uint8)
        self.spec_hash = np.zeros((num_envs,), dtype=np.uint64)


class FakeIdsOut:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, 4), dtype=np.int16)
        self.legal_ids = np.zeros((max(2, num_envs * 4),), dtype=np.uint32)
        self.legal_action_meta = np.full((max(2, num_envs * 4), 4), np.iinfo(np.uint16).max, dtype=np.uint16)
        self.legal_offsets = np.zeros((num_envs + 1,), dtype=np.int32)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=bool)
        self.truncated = np.zeros((num_envs,), dtype=bool)
        self.actor = np.zeros((num_envs,), dtype=np.int32)
        self.decision_kind = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int64)
        self.engine_status = np.zeros((num_envs,), dtype=np.uint8)
        self.spec_hash = np.zeros((num_envs,), dtype=np.uint64)


class ReadOnlyFakeIdsOut(FakeNoMaskOut):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self._legal_ids = np.zeros((max(2, num_envs * 4),), dtype=np.uint32)
        self._legal_action_meta = np.full((max(2, num_envs * 4), 4), np.iinfo(np.uint16).max, dtype=np.uint16)
        self._legal_offsets = np.zeros((num_envs + 1,), dtype=np.int32)

    @property
    def legal_ids(self) -> np.ndarray:
        return self._legal_ids

    @property
    def legal_offsets(self) -> np.ndarray:
        return self._legal_offsets

    @property
    def legal_action_meta(self) -> np.ndarray:
        return self._legal_action_meta


def copy_array(dst: Any, src: Any, *, allow_resize: bool = False) -> Any:
    src_array = np.asarray(src)
    needs_resize = not isinstance(dst, np.ndarray) or dst.shape != src_array.shape or dst.dtype != src_array.dtype
    if allow_resize and needs_resize:
        return src_array.copy()
    np.copyto(dst, src_array)
    return dst


COMMON_FAKE_STEP_FIELDS = (
    "obs",
    "rewards",
    "terminated",
    "truncated",
    "actor",
    "decision_kind",
    "decision_id",
    "engine_status",
    "spec_hash",
)


def copy_step_like(dst: Any, src: Any) -> None:
    for name in COMMON_FAKE_STEP_FIELDS:
        if hasattr(src, name):
            setattr(dst, name, copy_array(getattr(dst, name), getattr(src, name)))
    if hasattr(src, "masks"):
        dst.masks = copy_array(getattr(dst, "masks", None), src.masks, allow_resize=True)
    if hasattr(src, "legal_ids"):
        dst.legal_ids = copy_array(getattr(dst, "legal_ids", None), src.legal_ids, allow_resize=True)
    if hasattr(src, "legal_action_meta"):
        dst.legal_action_meta = copy_array(
            getattr(dst, "legal_action_meta", None),
            src.legal_action_meta,
            allow_resize=True,
        )
    if hasattr(src, "legal_offsets"):
        dst.legal_offsets = copy_array(
            getattr(dst, "legal_offsets", None),
            src.legal_offsets,
            allow_resize=True,
        )


class FakePool:
    def __init__(
        self,
        reported_rows: int | None = None,
        *,
        envs_len: int = 1,
        action_space: int = 52,
        mask_reset_result: object | None = None,
        nomask_reset_result: object | None = None,
        legal_ids_after_reset: np.ndarray | None = None,
        legal_action_meta_after_reset: np.ndarray | None = None,
        legal_offsets_after_reset: np.ndarray | None = None,
        episode_seed_batch: np.ndarray | None = None,
        episode_index_batch: np.ndarray | None = None,
        env_index_batch: np.ndarray | None = None,
    ) -> None:
        self.reported_rows = reported_rows
        self.envs_len = envs_len
        self.action_space = action_space
        self.mask_reset_result = mask_reset_result
        self.nomask_reset_result = nomask_reset_result
        self.legal_ids_after_reset = legal_ids_after_reset
        self.legal_action_meta_after_reset = legal_action_meta_after_reset
        self.legal_offsets_after_reset = legal_offsets_after_reset
        self._episode_seed_batch = (
            np.zeros((envs_len,), dtype=np.uint64) if episode_seed_batch is None else episode_seed_batch
        )
        self._episode_index_batch = (
            np.zeros((envs_len,), dtype=np.uint32) if episode_index_batch is None else episode_index_batch
        )
        self._env_index_batch = np.arange(envs_len, dtype=np.uint32) if env_index_batch is None else env_index_batch
        self.calls: list[tuple[str, np.ndarray, object]] = []
        self.seeded_reset_calls: list[tuple[str, np.ndarray, np.ndarray, object]] = []
        self.legal_refill_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.timing_enabled = False
        self.timing_reset_calls = 0
        self.timing_snapshot = {
            "timing_enabled": False,
            "step_sample_from_logits_with_logp_into_i16_legal_ids_count": 0,
            "step_sample_from_logits_with_logp_into_i16_legal_ids_ns": 0,
            "legal_action_meta_materialize_count": 0,
            "legal_action_meta_materialize_ns": 0,
        }

    def auto_reset_on_error_codes_into(self, codes: np.ndarray, out: Any) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append(("mask", copied_codes, out))
        if self.mask_reset_result is not None:
            copy_step_like(out, self.mask_reset_result)
        out.reset_applied = True
        return self.reported_rows

    def auto_reset_on_error_codes_into_nomask(self, codes: np.ndarray, out: Any) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append(("nomask", copied_codes, out))
        if self.nomask_reset_result is not None:
            copy_step_like(out, self.nomask_reset_result)
        out.reset_applied = True
        return self.reported_rows

    def reset_indices_with_episode_seeds_into(self, indices: list[int], episode_seeds: list[int], out: Any) -> None:
        self._reset_with_episode_seeds("mask", indices, episode_seeds, out)

    def reset_indices_with_episode_seeds_into_i16_legal_ids(
        self,
        indices: list[int],
        episode_seeds: list[int],
        out: Any,
    ) -> None:
        self._reset_with_episode_seeds("ids_offsets", indices, episode_seeds, out)

    def _reset_with_episode_seeds(
        self,
        method_name: str,
        indices: list[int],
        episode_seeds: list[int],
        out: Any,
    ) -> None:
        index_array = np.asarray(indices, dtype=np.int64)
        seed_array = np.asarray(episode_seeds, dtype=np.uint64)
        self._episode_seed_batch[index_array] = seed_array
        self.seeded_reset_calls.append((method_name, index_array.copy(), seed_array.copy(), out))

    def legal_action_ids_into(self, legal_ids: np.ndarray, legal_offsets: np.ndarray) -> int:
        if self.legal_ids_after_reset is None or self.legal_offsets_after_reset is None:
            raise AssertionError("legal_action_ids_into called without configured legal ids")
        legal_ids[...] = 0
        legal_ids[: self.legal_ids_after_reset.size] = self.legal_ids_after_reset
        legal_offsets[...] = 0
        legal_offsets[: self.legal_offsets_after_reset.size] = self.legal_offsets_after_reset
        self.legal_refill_calls.append((legal_ids.copy(), legal_offsets.copy()))
        return int(self.legal_ids_after_reset.size)

    def legal_action_meta_into(self, legal_action_meta: np.ndarray) -> int:
        legal_action_meta[...] = np.iinfo(legal_action_meta.dtype).max
        if self.legal_action_meta_after_reset is None:
            return 0
        legal_action_meta[: self.legal_action_meta_after_reset.shape[0]] = self.legal_action_meta_after_reset
        return int(self.legal_action_meta_after_reset.shape[0])

    def set_timing_enabled(self, enabled: bool) -> None:
        self.timing_enabled = bool(enabled)
        self.timing_snapshot["timing_enabled"] = bool(enabled)

    def reset_timing_counters(self) -> None:
        self.timing_reset_calls += 1

    def timing_counters(self) -> dict[str, int | bool]:
        return dict(self.timing_snapshot)

    def episode_seed_batch(self) -> np.ndarray:
        return self._episode_seed_batch

    def episode_index_batch(self) -> np.ndarray:
        return self._episode_index_batch

    def env_index_batch(self) -> np.ndarray:
        return self._env_index_batch


class FakeRlApi:
    def __init__(self, *, reset_result: object, step_result: object) -> None:
        self.reset_result = reset_result
        self.step_result = step_result
        self.reset_calls: list[tuple[object, str, object | None]] = []
        self.step_calls: list[tuple[object, np.ndarray, str, object | None]] = []

    def reset_rl(self, pool: object, *, layout: str, out: object | None = None) -> object:
        self.reset_calls.append((pool, layout, out))
        target = out if out is not None else self.reset_result
        copy_step_like(target, self.reset_result)
        return target

    def step_rl(self, pool: object, actions: np.ndarray, *, layout: str, out: object | None = None) -> object:
        self.step_calls.append((pool, np.asarray(actions).copy(), layout, out))
        target = out if out is not None else self.step_result
        copy_step_like(target, self.step_result)
        return target


class FakeWeissSim:
    PASS_ACTION_ID = 51
    BatchOutMinimal = FakeMaskOut
    BatchOutMinimalI16LegalIds = FakeIdsOut
    BatchOutMinimalNoMask = FakeNoMaskOut

    def __init__(self, *, reset_result: object, step_result: object) -> None:
        self.rl = FakeRlApi(reset_result=reset_result, step_result=step_result)


def load_weiss_sim():
    return pytest.importorskip("weiss_sim")


def simulator_episode_key(
    episode_seed: np.ndarray,
    episode_index: np.ndarray,
    env_index: np.ndarray,
) -> np.ndarray:
    runner = pytest.importorskip("weiss_sim.runner")
    return runner._episode_key(
        np.asarray(episode_seed, dtype=np.uint64),
        np.asarray(episode_index, dtype=np.uint64),
        np.asarray(env_index, dtype=np.uint64),
    )


def make_env(*, legality: LegalMode, num_envs: int = 2, seed: int = 123) -> DecisionBoundaryEnv:
    load_weiss_sim()
    return DecisionBoundaryEnv.create(
        legality=legality,
        mode="train",
        num_envs=num_envs,
        db_path=None,
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=seed,
    )


def assert_batch_episode_identity_matches_pool(batch: DecisionBoundaryBatch, pool: Any) -> None:
    episode_seed = pool.episode_seed_batch().astype(np.uint64, copy=False)
    episode_index = pool.episode_index_batch().astype(np.uint64, copy=False)
    env_index = pool.env_index_batch().astype(np.uint64, copy=False)
    expected_episode_key = simulator_episode_key(episode_seed, episode_index, env_index)
    npt.assert_array_equal(batch.episode_seed, episode_seed)
    npt.assert_array_equal(batch.episode_key, expected_episode_key)


def first_legal_from_mask(mask: np.ndarray, *, pass_action_id: int) -> np.ndarray:
    actions = np.empty((int(mask.shape[0]),), dtype=np.uint32)
    for env_index in range(mask.shape[0]):
        legal = np.flatnonzero(mask[env_index])
        actions[env_index] = pass_action_id if legal.size == 0 else int(legal[0])
    return actions


def first_legal_from_ids(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int,
) -> np.ndarray:
    num_envs = int(legal_offsets.shape[0]) - 1
    actions = np.empty((num_envs,), dtype=np.uint32)
    for env_index in range(num_envs):
        start = int(legal_offsets[env_index])
        end = int(legal_offsets[env_index + 1])
        actions[env_index] = pass_action_id if start == end else int(legal_ids[start])
    return actions


def illegal_action_from_mask(mask: np.ndarray) -> int:
    illegal = np.flatnonzero(~mask[0].astype(bool))
    if illegal.size == 0:
        raise AssertionError("expected at least one illegal action")
    return int(illegal[0])


def illegal_action_from_ids(legal_ids: np.ndarray, legal_offsets: np.ndarray, *, action_space: int) -> int:
    start = int(legal_offsets[0])
    end = int(legal_offsets[1])
    legal = set(int(action) for action in legal_ids[start:end].tolist())
    for action in range(action_space):
        if action not in legal:
            return action
    raise AssertionError("expected at least one illegal action")


def done_step(legality: LegalMode) -> SimpleNamespace:
    step_kwargs = {
        "obs": np.zeros((1, 4), dtype=np.float32),
        "rewards": np.zeros((1,), dtype=np.float32),
        "terminated": np.array([False]),
        "truncated": np.array([True]),
        "actor": np.array([-1], dtype=np.int32),
        "decision_kind": np.array([-1], dtype=np.int32),
        "decision_id": np.array([17], dtype=np.int64),
        "engine_status": np.array([0], dtype=np.uint8),
        "spec_hash": np.array([0], dtype=np.uint64),
    }
    if legality == "mask":
        return SimpleNamespace(**step_kwargs, masks=np.zeros((1, 52), dtype=np.uint8))
    step_kwargs["obs"] = np.zeros((1, 4), dtype=np.int16)
    return SimpleNamespace(
        **step_kwargs,
        legal_ids=np.array([], dtype=np.uint32),
        legal_offsets=np.array([0, 0], dtype=np.int32),
    )


def mask_step(*, engine_status: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        rewards=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([0], dtype=np.int32),
        decision_kind=np.array([0], dtype=np.int32),
        decision_id=np.array([17], dtype=np.int64),
        engine_status=np.asarray(engine_status, dtype=np.uint8),
        spec_hash=np.array([0], dtype=np.uint64),
        masks=np.ones((1, 52), dtype=np.uint8),
    )


def ids_step(*, engine_status: np.ndarray, reward: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.int16),
        rewards=np.array([reward], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([0], dtype=np.int32),
        decision_kind=np.array([0], dtype=np.int32),
        decision_id=np.array([17], dtype=np.int64),
        engine_status=np.asarray(engine_status, dtype=np.uint8),
        spec_hash=np.array([0], dtype=np.uint64),
        legal_ids=np.array([1, 5], dtype=np.uint32),
        legal_offsets=np.array([0, 2], dtype=np.int32),
    )
