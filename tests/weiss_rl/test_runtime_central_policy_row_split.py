from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
from weiss_rl.runtime import QueueRuntime

from .runtime_central_rows_test_support import FixedRng, bare_queue_runtime


def test_central_sample_policy_rows_routes_fractional_heuristic_and_model_rows() -> None:
    runtime = bare_queue_runtime()
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 0.5

    calls: list[tuple[str, list[np.ndarray]]] = []

    def record_model(**kwargs: Any) -> None:
        calls.append(("model", [np.array(rows, copy=True) for rows in kwargs["row_indices_by_actor"]]))

    def record_heuristic(**kwargs: Any) -> None:
        calls.append(("heuristic", [np.array(rows, copy=True) for rows in kwargs["row_indices_by_actor"]]))

    runtime_any._central_sample_policy_rows_ids_model = record_model
    runtime_any._central_sample_policy_rows_ids_heuristic = record_heuristic

    actors = [
        SimpleNamespace(rng=FixedRng((0.1, 0.9, 0.4))),
        SimpleNamespace(rng=FixedRng((0.6, 0.2))),
        SimpleNamespace(rng=FixedRng(())),
    ]
    row_indices_by_actor = [
        np.asarray([2, 4, 6], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        np.asarray([], dtype=np.int64),
    ]

    QueueRuntime._central_sample_policy_rows_ids(
        runtime,
        actors=cast(Any, actors),
        batches=[object(), object(), object()],
        obs_steps=[np.empty((0, 1), dtype=np.float32)] * 3,
        actor_steps=[np.empty((0,), dtype=np.int64)] * 3,
        row_indices_by_actor=row_indices_by_actor,
        values_outs=[np.empty((0,), dtype=np.float32)] * 3,
        actions_outs=[np.empty((0,), dtype=np.int64)] * 3,
        logp_outs=[np.empty((0,), dtype=np.float32)] * 3,
    )

    assert [label for label, _rows in calls] == ["heuristic", "model"]
    heuristic_rows = calls[0][1]
    model_rows = calls[1][1]
    npt.assert_array_equal(heuristic_rows[0], np.asarray([2, 6], dtype=np.int64))
    npt.assert_array_equal(heuristic_rows[1], np.asarray([3], dtype=np.int64))
    npt.assert_array_equal(heuristic_rows[2], np.asarray([], dtype=np.int64))
    npt.assert_array_equal(model_rows[0], np.asarray([4], dtype=np.int64))
    npt.assert_array_equal(model_rows[1], np.asarray([1], dtype=np.int64))
    npt.assert_array_equal(model_rows[2], np.asarray([], dtype=np.int64))
