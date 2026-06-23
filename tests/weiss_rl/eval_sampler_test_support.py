from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from weiss_rl.core.masking import masked_logp_from_legal_ids

TEST_VECTORS_PATH = Path(__file__).with_name("test_vectors") / "pcg32_xsh_rr_v1.json"
PINNED_VECTOR_SEEDS = [
    "0",
    "1",
    "42",
    "20260316",
    "16045690984503098046",
    "18446744073709551615",
]
SEEDED_STREAM_ANCHOR: StreamAnchor = {
    "seed64": 20260316,
    "uint32_outputs": [
        1819242021,
        1579062260,
        3226218236,
        3871637039,
        2919469525,
        360479254,
        278646790,
        3491948883,
        2582750945,
        2072432381,
    ],
    "uint64_outputs": [
        "0x6c6f6e255e1e93f4",
        "0xc04c2efce6c47e2f",
        "0xae0391d5157c7a16",
        "0x109bd006d022e953",
        "0x99f1a6e17b86cefd",
    ],
    "uniform01_hex": [
        "0x1.b1bdb895787a5p-2",
        "0x1.80985df9cd890p-1",
        "0x1.5c0723aa2af90p-1",
        "0x1.09bd006d022ecp-4",
        "0x1.33e34dc2f70dap-1",
    ],
}
PINNED_SAMPLER_ACTIONS = [4, 5, 5, 0, 5, 5, 7, 7, 4, 4]


class VectorCase(TypedDict):
    seed64: str
    uint32_outputs: list[int]
    uint64_outputs: list[str]
    uniform01_hex: list[str]


class VectorPayload(TypedDict):
    algorithm: str
    next_u64_order: str
    float_conversion: str
    cases: list[VectorCase]


class StreamAnchor(TypedDict):
    seed64: int
    uint32_outputs: list[int]
    uint64_outputs: list[str]
    uniform01_hex: list[str]


class StubFloatRng:
    def __init__(self, *draws: float) -> None:
        self._draws = list(draws)
        self.calls = 0

    def next_float(self) -> float:
        if self.calls >= len(self._draws):
            raise AssertionError("stub rng exhausted")
        draw = self._draws[self.calls]
        self.calls += 1
        return draw


def load_vectors() -> VectorPayload:
    payload = json.loads(TEST_VECTORS_PATH.read_text(encoding="utf-8"))
    return cast(VectorPayload, payload)


def vector_cases() -> list[VectorCase]:
    return load_vectors()["cases"]


def expected_single_row_logp(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    action: int,
    *,
    pass_action_id: int | None = None,
    temperature: float = 1.0,
) -> np.float32:
    scaled_logits = logits / np.float32(temperature)
    legal_offsets = np.array([0, legal_ids.size], dtype=np.int64)
    actions = np.array([action], dtype=np.int64)
    logp = masked_logp_from_legal_ids(
        scaled_logits[np.newaxis, :],
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )
    return np.float32(logp[0])
