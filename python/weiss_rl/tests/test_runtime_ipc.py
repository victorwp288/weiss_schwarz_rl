from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from weiss_rl.runtime.components.ipc import deserialize_state_dict_from_ipc, serialize_state_dict_for_ipc


def test_serialize_state_dict_for_ipc_converts_tensors_to_independent_numpy_arrays() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    metadata = {"nested": [1, 2]}
    serialized = serialize_state_dict_for_ipc({"weight": tensor, "metadata": metadata})

    assert isinstance(serialized["weight"], np.ndarray)
    assert serialized["weight"].dtype == np.float32
    assert serialized["weight"].tolist() == [[1.0, 2.0]]
    assert serialized["metadata"] == {"nested": [1, 2]}

    with torch.no_grad():
        tensor[0, 0] = 99.0
    metadata["nested"].append(3)

    assert serialized["weight"].tolist() == [[1.0, 2.0]]
    assert serialized["metadata"] == {"nested": [1, 2]}


def test_deserialize_state_dict_from_ipc_converts_numpy_arrays_to_independent_tensors() -> None:
    array = np.array([[3, 4]], dtype=np.int64)
    metadata = {"nested": [5, 6]}
    restored = deserialize_state_dict_from_ipc({"weight": array, "metadata": metadata})

    assert isinstance(restored["weight"], torch.Tensor)
    assert restored["weight"].dtype == torch.int64
    assert restored["weight"].tolist() == [[3, 4]]
    assert restored["metadata"] == {"nested": [5, 6]}

    array[0, 0] = 99
    metadata["nested"].append(7)

    assert restored["weight"].tolist() == [[3, 4]]
    assert restored["metadata"] == {"nested": [5, 6]}


def test_state_dict_ipc_round_trip_preserves_stringified_keys_and_tensor_values() -> None:
    restored = deserialize_state_dict_from_ipc(
        serialize_state_dict_for_ipc(cast(Any, {7: torch.tensor([1.25], dtype=torch.float32)}))
    )

    assert sorted(restored) == ["7"]
    assert restored["7"].tolist() == [1.25]
