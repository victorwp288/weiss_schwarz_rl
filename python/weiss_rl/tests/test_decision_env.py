from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import (
    DecisionBoundaryBatch,
    DecisionBoundaryEnv,
    EngineStatusCounters,
    LegalMode,
    _derive_episode_key,
    _merge_packed_legality_rows,
    _pack_batch,
    _validate_actions,
)

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
        self._legal_offsets = np.zeros((num_envs + 1,), dtype=np.int32)

    @property
    def legal_ids(self) -> np.ndarray:
        return self._legal_ids

    @property
    def legal_offsets(self) -> np.ndarray:
        return self._legal_offsets


def _copy_array(dst: Any, src: Any, *, allow_resize: bool = False) -> Any:
    src_array = np.asarray(src)
    needs_resize = not isinstance(dst, np.ndarray) or dst.shape != src_array.shape or dst.dtype != src_array.dtype
    if allow_resize and needs_resize:
        return src_array.copy()
    np.copyto(dst, src_array)
    return dst


_COMMON_FAKE_STEP_FIELDS = (
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


def _copy_step_like(dst: Any, src: Any) -> None:
    for name in _COMMON_FAKE_STEP_FIELDS:
        if hasattr(src, name):
            setattr(dst, name, _copy_array(getattr(dst, name), getattr(src, name)))
    if hasattr(src, "masks"):
        dst.masks = _copy_array(getattr(dst, "masks", None), src.masks, allow_resize=True)
    if hasattr(src, "legal_ids"):
        dst.legal_ids = _copy_array(getattr(dst, "legal_ids", None), src.legal_ids, allow_resize=True)
    if hasattr(src, "legal_offsets"):
        dst.legal_offsets = _copy_array(
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

    def auto_reset_on_error_codes_into(self, codes: np.ndarray, out: Any) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append(("mask", copied_codes, out))
        if self.mask_reset_result is not None:
            _copy_step_like(out, self.mask_reset_result)
        out.reset_applied = True
        return self.reported_rows

    def auto_reset_on_error_codes_into_nomask(self, codes: np.ndarray, out: Any) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append(("nomask", copied_codes, out))
        if self.nomask_reset_result is not None:
            _copy_step_like(out, self.nomask_reset_result)
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
        _copy_step_like(target, self.reset_result)
        return target

    def step_rl(self, pool: object, actions: np.ndarray, *, layout: str, out: object | None = None) -> object:
        self.step_calls.append((pool, np.asarray(actions).copy(), layout, out))
        target = out if out is not None else self.step_result
        _copy_step_like(target, self.step_result)
        return target


class FakeWeissSim:
    PASS_ACTION_ID = 51
    BatchOutMinimal = FakeMaskOut
    BatchOutMinimalI16LegalIds = FakeIdsOut
    BatchOutMinimalNoMask = FakeNoMaskOut

    def __init__(self, *, reset_result: object, step_result: object) -> None:
        self.rl = FakeRlApi(reset_result=reset_result, step_result=step_result)


def _load_weiss_sim():
    return pytest.importorskip("weiss_sim")


def _simulator_episode_key(
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


def _make_env(*, legality: LegalMode, num_envs: int = 2, seed: int = 123) -> DecisionBoundaryEnv:
    _load_weiss_sim()
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


def _assert_batch_episode_identity_matches_pool(batch: DecisionBoundaryBatch, pool: Any) -> None:
    episode_seed = pool.episode_seed_batch().astype(np.uint64, copy=False)
    episode_index = pool.episode_index_batch().astype(np.uint64, copy=False)
    env_index = pool.env_index_batch().astype(np.uint64, copy=False)
    expected_episode_key = _simulator_episode_key(episode_seed, episode_index, env_index)
    npt.assert_array_equal(batch.episode_seed, episode_seed)
    npt.assert_array_equal(batch.episode_key, expected_episode_key)


def _first_legal_from_mask(mask: np.ndarray, *, pass_action_id: int) -> np.ndarray:
    actions = np.empty((int(mask.shape[0]),), dtype=np.uint32)
    for env_index in range(mask.shape[0]):
        legal = np.flatnonzero(mask[env_index])
        actions[env_index] = pass_action_id if legal.size == 0 else int(legal[0])
    return actions


def _first_legal_from_ids(
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


def _illegal_action_from_mask(mask: np.ndarray) -> int:
    illegal = np.flatnonzero(~mask[0].astype(bool))
    if illegal.size == 0:
        raise AssertionError("expected at least one illegal action")
    return int(illegal[0])


def _illegal_action_from_ids(legal_ids: np.ndarray, legal_offsets: np.ndarray, *, action_space: int) -> int:
    start = int(legal_offsets[0])
    end = int(legal_offsets[1])
    legal = set(int(action) for action in legal_ids[start:end].tolist())
    for action in range(action_space):
        if action not in legal:
            return action
    raise AssertionError("expected at least one illegal action")


def _done_step(legality: LegalMode) -> SimpleNamespace:
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


def _mask_step(*, engine_status: np.ndarray) -> SimpleNamespace:
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


def _ids_step(*, engine_status: np.ndarray, reward: float = 0.0) -> SimpleNamespace:
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


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_done_batch_with_actor_minus_one_and_empty_legality_is_pass_only(legality: LegalMode) -> None:
    pass_action_id = 51
    batch = _pack_batch(_done_step(legality), legality=legality)

    assert int(batch.actor[0]) == -1
    assert bool(batch.truncated[0])

    _validate_actions(np.array([pass_action_id], dtype=np.uint32), batch, pass_action_id=pass_action_id)
    with pytest.raises(ValueError, match=f"expected pass action {pass_action_id}"):
        _validate_actions(np.array([0], dtype=np.uint32), batch, pass_action_id=pass_action_id)


def test_pack_batch_ids_offsets_trims_raw_capacity_and_derives_episode_identity_from_pool() -> None:
    step = SimpleNamespace(
        obs=np.zeros((2, 4), dtype=np.int16),
        rewards=np.zeros((2,), dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([0, 1], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([7, 8], dtype=np.int64),
        engine_status=np.array([0, 0], dtype=np.uint8),
        spec_hash=np.array([11, 12], dtype=np.uint64),
        legal_ids=np.array([1, 2, 3, 4, 99, 98], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.uint32),
    )
    pool = FakePool(
        envs_len=2,
        episode_seed_batch=np.array([101, 202], dtype=np.uint64),
        episode_index_batch=np.array([5, 6], dtype=np.uint32),
        env_index_batch=np.array([0, 1], dtype=np.uint32),
    )

    batch = _pack_batch(step, legality="ids_offsets", pool=pool)

    assert batch.ids_offsets is not None
    legal_ids, legal_offsets = batch.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([1, 2, 3, 4], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2, 4], dtype=np.uint32))
    npt.assert_array_equal(batch.episode_seed, np.array([101, 202], dtype=np.uint64))
    npt.assert_array_equal(
        batch.episode_key,
        _simulator_episode_key(
            np.array([101, 202], dtype=np.uint64),
            np.array([5, 6], dtype=np.uint64),
            np.array([0, 1], dtype=np.uint64),
        ),
    )


def test_derive_episode_key_matches_weiss_sim_runner_helper() -> None:
    episode_seed = np.array([0, 1, 2, 17, 123456789, 2**64 - 1], dtype=np.uint64)
    episode_index = np.array([0, 1, 7, 99, 2**16, 2**32 - 1], dtype=np.uint64)
    env_index = np.array([0, 1, 2, 3, 255, 2**32 - 1], dtype=np.uint64)

    actual = _derive_episode_key(episode_seed, episode_index, env_index)
    expected = _simulator_episode_key(episode_seed, episode_index, env_index)

    npt.assert_array_equal(actual, expected)


def test_decision_env_create_uses_lazy_weiss_sim_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("weiss_rl.envs.decision_env")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "weiss_sim":
            raise ImportError("missing simulator")
        return real_import_module(name, package)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="weiss_sim"):
        module.DecisionBoundaryEnv.create(legality="mask", mode="train", num_envs=1)


def test_mask_mode_reset_and_step_return_typed_batch() -> None:
    env = _make_env(legality="mask")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is not None
        assert batch.ids_offsets is None
        assert batch.obs.shape[0] == env.num_envs
        assert batch.reward.shape == (env.num_envs,)
        assert batch.terminated.shape == (env.num_envs,)
        assert batch.truncated.shape == (env.num_envs,)
        assert batch.decision_id.shape == (env.num_envs,)
        assert batch.engine_status.shape == (env.num_envs,)
        assert batch.episode_seed.shape == (env.num_envs,)
        assert batch.episode_key.shape == (env.num_envs,)
        assert np.array_equal(batch.to_play, batch.actor)
        _assert_batch_episode_identity_matches_pool(batch, env.pool)

        next_batch = env.step(_first_legal_from_mask(batch.mask, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is not None
        assert next_batch.ids_offsets is None
        assert next_batch.obs.shape[0] == env.num_envs
        assert next_batch.reward.shape == (env.num_envs,)
        _assert_batch_episode_identity_matches_pool(next_batch, env.pool)
    finally:
        env.close()


def test_ids_offsets_mode_reset_and_step_return_typed_batch() -> None:
    env = _make_env(legality="ids_offsets")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is None
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        assert legal_ids.ndim == 1
        assert legal_offsets.shape == (env.num_envs + 1,)
        assert legal_ids.shape == (int(legal_offsets[-1]),)
        assert batch.episode_seed.shape == (env.num_envs,)
        assert batch.episode_key.shape == (env.num_envs,)
        assert np.array_equal(batch.to_play, batch.actor)
        _assert_batch_episode_identity_matches_pool(batch, env.pool)

        next_batch = env.step(_first_legal_from_ids(legal_ids, legal_offsets, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is None
        assert next_batch.ids_offsets is not None
        next_legal_ids, next_legal_offsets = next_batch.ids_offsets
        assert next_legal_ids.shape == (int(next_legal_offsets[-1]),)
        assert next_batch.reward.shape == (env.num_envs,)
        _assert_batch_episode_identity_matches_pool(next_batch, env.pool)
    finally:
        env.close()


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_step_rejects_illegal_action(legality: LegalMode) -> None:
    env = _make_env(legality=legality, num_envs=1)
    try:
        batch = env.reset()
        if batch.mask is not None:
            illegal_action = _illegal_action_from_mask(batch.mask)
        else:
            assert batch.ids_offsets is not None
            legal_ids, legal_offsets = batch.ids_offsets
            illegal_action = _illegal_action_from_ids(
                legal_ids,
                legal_offsets,
                action_space=env.action_space,
            )

        with pytest.raises(ValueError, match="illegal action"):
            env.step(np.array([illegal_action], dtype=np.uint32))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("legality", "expected_method"),
    [("mask", "mask"), ("ids_offsets", "ids_offsets")],
)
def test_reset_with_explicit_seed_uses_pool_seeded_reset_contract(
    legality: LegalMode,
    expected_method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool(envs_len=2)
    if legality == "mask":
        reset_step = SimpleNamespace(
            obs=np.zeros((2, 4), dtype=np.float32),
            rewards=np.zeros((2,), dtype=np.float32),
            terminated=np.zeros((2,), dtype=bool),
            truncated=np.zeros((2,), dtype=bool),
            actor=np.zeros((2,), dtype=np.int32),
            decision_kind=np.zeros((2,), dtype=np.int32),
            decision_id=np.zeros((2,), dtype=np.int64),
            engine_status=np.zeros((2,), dtype=np.uint8),
            spec_hash=np.zeros((2,), dtype=np.uint64),
            masks=np.zeros((2, 52), dtype=np.uint8),
        )
    else:
        reset_step = SimpleNamespace(
            obs=np.zeros((2, 4), dtype=np.int16),
            rewards=np.zeros((2,), dtype=np.float32),
            terminated=np.zeros((2,), dtype=bool),
            truncated=np.zeros((2,), dtype=bool),
            actor=np.zeros((2,), dtype=np.int32),
            decision_kind=np.zeros((2,), dtype=np.int32),
            decision_id=np.zeros((2,), dtype=np.int64),
            engine_status=np.zeros((2,), dtype=np.uint8),
            spec_hash=np.zeros((2,), dtype=np.uint64),
            legal_ids=np.array([], dtype=np.uint32),
            legal_offsets=np.array([0, 0, 0], dtype=np.int32),
        )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=reset_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality=legality)
    batch = env.reset(seed=77)

    assert fake_weiss_sim.rl.reset_calls == []
    assert len(pool.seeded_reset_calls) == 1
    method_name, indices, episode_seeds, out = pool.seeded_reset_calls[0]
    assert method_name == expected_method
    npt.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    npt.assert_array_equal(episode_seeds, np.array([77, 77], dtype=np.uint64))
    npt.assert_array_equal(batch.episode_seed, np.array([77, 77], dtype=np.uint64))
    assert out is not None


def test_hard_fail_raises_and_counts_fault_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool()
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = _mask_step(engine_status=np.array([7], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, engine_status_policy="hard_fail", counters=counters)
    env.reset()

    with pytest.raises(RuntimeError, match=r"fault_rows=1"):
        env.step(np.array([10], dtype=np.int64))

    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert len(fake_weiss_sim.rl.step_calls) == 1


def test_passthrough_preserves_faulted_step_without_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool()
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = _mask_step(engine_status=np.array([7], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, engine_status_policy="passthrough", counters=counters)
    env.reset()
    returned = env.step(np.array([10], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert pool.calls == []
    npt.assert_array_equal(returned.engine_status, np.array([7], dtype=np.uint8))


def test_best_effort_reset_mask_uses_pool_contract_and_counts_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    mask_reset = _mask_step(engine_status=np.array([0], dtype=np.uint8))
    pool = FakePool(reported_rows=1, mask_reset_result=mask_reset)
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = _mask_step(engine_status=np.array([3], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([1], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    assert len(pool.calls) == 1
    method_name, reset_codes, reset_out = pool.calls[0]
    assert method_name == "mask"
    assert isinstance(reset_out, FakeMaskOut)
    assert getattr(reset_out, "reset_applied", False) is True
    assert reset_codes.dtype == np.uint8
    assert reset_codes.flags.c_contiguous
    npt.assert_array_equal(reset_codes, np.array([3], dtype=np.uint8))
    npt.assert_array_equal(returned.engine_status, np.array([0], dtype=np.uint8))


def test_best_effort_reset_ids_offsets_uses_nomask_contract_and_refills_legal_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = EngineStatusCounters()
    reset_step = _ids_step(engine_status=np.array([0], dtype=np.uint8))
    faulted_step = _ids_step(engine_status=np.array([5], dtype=np.uint8))
    nomask_reset = SimpleNamespace(
        obs=np.array([[7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
        rewards=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([1], dtype=np.int32),
        decision_kind=np.array([0], dtype=np.int32),
        decision_id=np.array([21], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        spec_hash=np.array([11], dtype=np.uint64),
    )
    pool = FakePool(
        reported_rows=1,
        nomask_reset_result=nomask_reset,
        legal_ids_after_reset=np.array([2, 4], dtype=np.uint32),
        legal_offsets_after_reset=np.array([0, 2], dtype=np.int32),
    )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=faulted_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality="ids_offsets", counters=counters)
    env.reset()
    returned = env.step(np.array([1], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    assert len(pool.calls) == 1
    method_name, reset_codes, reset_out = pool.calls[0]
    assert method_name == "nomask"
    assert isinstance(reset_out, FakeNoMaskOut)
    assert getattr(reset_out, "reset_applied", False) is True
    npt.assert_array_equal(reset_codes, np.array([5], dtype=np.uint8))
    assert len(pool.legal_refill_calls) == 1
    assert returned.ids_offsets is not None
    legal_ids, legal_offsets = returned.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([2, 4], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2], dtype=np.int32))
    npt.assert_array_equal(returned.actor, np.array([1], dtype=np.int32))
    npt.assert_array_equal(returned.engine_status, np.array([0], dtype=np.uint8))
    npt.assert_array_equal(returned.obs, np.array([[7, 8, 9, 10]], dtype=np.int16))


def test_best_effort_reset_ids_offsets_only_replaces_fault_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    reset_step = SimpleNamespace(
        obs=np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int16),
        rewards=np.array([0.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([0, 1], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([10, 20], dtype=np.int64),
        engine_status=np.array([0, 0], dtype=np.uint8),
        spec_hash=np.array([11, 22], dtype=np.uint64),
        legal_ids=np.array([1, 5, 7, 8], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.int32),
    )
    faulted_step = SimpleNamespace(
        obs=np.array([[3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.int16),
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([3, 4], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([30, 40], dtype=np.int64),
        engine_status=np.array([0, 9], dtype=np.uint8),
        spec_hash=np.array([33, 44], dtype=np.uint64),
        legal_ids=np.array([1, 5, 7, 8], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.int32),
    )
    nomask_reset = SimpleNamespace(
        obs=np.array([[99.0, 99.0, 99.0, 99.0], [8.0, 8.0, 8.0, 8.0]], dtype=np.float32),
        rewards=np.array([9.0, 0.0], dtype=np.float32),
        terminated=np.array([True, False]),
        truncated=np.array([True, False]),
        actor=np.array([9, 6], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([99, 60], dtype=np.int64),
        engine_status=np.array([7, 0], dtype=np.uint8),
        spec_hash=np.array([99, 66], dtype=np.uint64),
    )
    pool = FakePool(
        reported_rows=1,
        envs_len=2,
        nomask_reset_result=nomask_reset,
        legal_ids_after_reset=np.array([9, 10, 11, 12], dtype=np.uint32),
        legal_offsets_after_reset=np.array([0, 2, 4], dtype=np.int32),
    )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=faulted_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality="ids_offsets", counters=counters)
    env.reset()
    returned = env.step(np.array([1, 7], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    npt.assert_array_equal(returned.obs, np.array([[3, 3, 3, 3], [8, 8, 8, 8]], dtype=np.int16))
    npt.assert_array_equal(returned.reward, np.array([1.0, 0.0], dtype=np.float32))
    npt.assert_array_equal(returned.actor, np.array([3, 6], dtype=np.int32))
    npt.assert_array_equal(returned.decision_id, np.array([30, 60], dtype=np.int64))
    npt.assert_array_equal(returned.engine_status, np.array([0, 0], dtype=np.uint8))
    assert returned.ids_offsets is not None
    legal_ids, legal_offsets = returned.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([1, 5, 11, 12], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2, 4], dtype=np.int32))


def test_merge_packed_legality_rows_mutates_read_only_fake_buffers_in_place() -> None:
    current = ReadOnlyFakeIdsOut(2)
    replacement = ReadOnlyFakeIdsOut(2)
    current.legal_ids[:6] = np.array([1, 5, 7, 8, 99, 99], dtype=np.uint32)
    current.legal_offsets[:] = np.array([0, 2, 4], dtype=np.int32)
    replacement.legal_ids[:4] = np.array([9, 10, 11, 12], dtype=np.uint32)
    replacement.legal_offsets[:] = np.array([0, 2, 4], dtype=np.int32)

    ids_before = current.legal_ids
    offsets_before = current.legal_offsets
    _merge_packed_legality_rows(
        dst=current,
        current=current,
        replacement=replacement,
        rows=np.array([False, True]),
    )

    assert current.legal_ids is ids_before
    assert current.legal_offsets is offsets_before
    npt.assert_array_equal(current.legal_offsets, np.array([0, 2, 4], dtype=np.int32))
    npt.assert_array_equal(current.legal_ids[:4], np.array([1, 5, 11, 12], dtype=np.uint32))
    npt.assert_array_equal(current.legal_ids[4:6], np.array([0, 0], dtype=np.uint32))


def test_merge_packed_legality_rows_mutates_real_sim_buffers_in_place() -> None:
    weiss_sim = pytest.importorskip("weiss_sim")
    current = weiss_sim.BatchOutMinimalI16LegalIds(2)
    replacement = weiss_sim.BatchOutMinimalI16LegalIds(2)
    current.legal_ids[:6] = np.array([1, 5, 7, 8, 99, 99], dtype=current.legal_ids.dtype)
    current.legal_offsets[:] = np.array([0, 2, 4], dtype=current.legal_offsets.dtype)
    replacement.legal_ids[:4] = np.array([9, 10, 11, 12], dtype=replacement.legal_ids.dtype)
    replacement.legal_offsets[:] = np.array([0, 2, 4], dtype=replacement.legal_offsets.dtype)

    ids_before = current.legal_ids
    offsets_before = current.legal_offsets
    _merge_packed_legality_rows(
        dst=current,
        current=current,
        replacement=replacement,
        rows=np.array([False, True]),
    )

    assert current.legal_ids is ids_before
    assert current.legal_offsets is offsets_before
    npt.assert_array_equal(current.legal_offsets, np.array([0, 2, 4], dtype=current.legal_offsets.dtype))
    npt.assert_array_equal(current.legal_ids[:4], np.array([1, 5, 11, 12], dtype=current.legal_ids.dtype))
    npt.assert_array_equal(current.legal_ids[4:6], np.array([0, 0], dtype=current.legal_ids.dtype))


def test_best_effort_reset_is_a_noop_when_pool_support_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = SimpleNamespace(envs_len=1, action_space=52)
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = _mask_step(engine_status=np.array([4], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([9], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_pack_batch_returns_snapshot_not_sim_buffer_view(legality: LegalMode) -> None:
    step = (
        _done_step(legality)
        if legality == "mask"
        else _ids_step(engine_status=np.array([0], dtype=np.uint8), reward=0.0)
    )
    batch = _pack_batch(step, legality=legality)

    step.obs[...] = 77
    step.rewards[...] = 12
    step.terminated[...] = True
    step.truncated[...] = False
    step.actor[...] = 5
    step.decision_id[...] = 99
    step.engine_status[...] = 3
    if legality == "mask":
        step.masks[...] = 1
        assert batch.mask is not None
        assert not np.array_equal(batch.mask, step.masks)
    else:
        step.legal_ids[...] = 4
        step.legal_offsets[...] = 1
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        assert not np.array_equal(legal_ids, step.legal_ids)
        assert not np.array_equal(legal_offsets, step.legal_offsets)

    assert not np.array_equal(batch.obs, step.obs)
    assert not np.array_equal(batch.reward, step.rewards)
    assert not np.array_equal(batch.terminated, step.terminated)
    assert not np.array_equal(batch.actor, step.actor)


def test_create_threads_engine_status_policy_and_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = FakePool(envs_len=2, action_space=52)
    fake_weiss_sim = SimpleNamespace(
        PASS_ACTION_ID=51,
        make_pool=lambda **kwargs: (pool, {"kwargs": kwargs}),
    )
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    counters = EngineStatusCounters()
    env = DecisionBoundaryEnv.create(
        legality="mask",
        engine_status_policy="hard_fail",
        counters=counters,
        mode="train",
        num_envs=2,
    )

    assert env.pool is pool
    assert env.pass_action_id == 51
    assert env.engine_status_policy == "hard_fail"
    assert env.counters is counters
