from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import pytest

from weiss_rl.eval.rng_pcg32 import NEXT_U64_ORDER, PCG32_XSH_RR_V1, Pcg32XshRrV1

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


def _load_vectors() -> VectorPayload:
    payload = json.loads(TEST_VECTORS_PATH.read_text(encoding="utf-8"))
    return cast(VectorPayload, payload)


def _cases() -> list[VectorCase]:
    return _load_vectors()["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_u32_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [rng.next_u32() for _ in range(10)]
    assert outputs == case["uint32_outputs"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_u64_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [f"0x{rng.next_u64():016x}" for _ in range(5)]
    assert outputs == case["uint64_outputs"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_float_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [rng.next_float().hex() for _ in range(5)]
    assert outputs == case["uniform01_hex"]


def test_pcg32_vector_corpus_shape_is_pinned() -> None:
    cases = _cases()
    assert [case["seed64"] for case in cases] == PINNED_VECTOR_SEEDS
    assert len(cases) >= 5
    assert len({case["seed64"] for case in cases}) == len(cases)
    for case in cases:
        assert len(case["uint32_outputs"]) == 10
        assert len(case["uint64_outputs"]) == 5
        assert len(case["uniform01_hex"]) == 5


def test_pcg32_seeded_stream_matches_anchor_case() -> None:
    seed64 = SEEDED_STREAM_ANCHOR["seed64"]

    u32_rng = Pcg32XshRrV1(seed64)
    assert [u32_rng.next_u32() for _ in range(10)] == SEEDED_STREAM_ANCHOR["uint32_outputs"]

    u64_rng = Pcg32XshRrV1(seed64)
    assert [f"0x{u64_rng.next_u64():016x}" for _ in range(5)] == SEEDED_STREAM_ANCHOR["uint64_outputs"]

    float_rng = Pcg32XshRrV1(seed64)
    assert [float_rng.next_float().hex() for _ in range(5)] == SEEDED_STREAM_ANCHOR["uniform01_hex"]


def test_pcg32_metadata_is_pinned() -> None:
    payload = _load_vectors()
    assert payload["algorithm"] == PCG32_XSH_RR_V1
    assert payload["next_u64_order"] == NEXT_U64_ORDER
    assert payload["float_conversion"] == "u = ((x >> 11) + 0.5) / 2**53"


def test_pcg32_next_u64_uses_hi_then_lo_concatenation() -> None:
    seed64 = 20260316
    left = Pcg32XshRrV1(seed64)
    hi = left.next_u32()
    lo = left.next_u32()

    right = Pcg32XshRrV1(seed64)
    assert right.next_u64() == ((hi << 32) | lo)


def test_pcg32_next_float_uses_top_53_bits() -> None:
    seed64 = 20260316
    left = Pcg32XshRrV1(seed64)
    x = left.next_u64()

    right = Pcg32XshRrV1(seed64)
    expected = ((x >> 11) + 0.5) / float(1 << 53)
    assert right.next_float() == expected


def test_pcg32_rejects_out_of_range_seed() -> None:
    with pytest.raises(ValueError, match="rng_seed64 must be in"):
        Pcg32XshRrV1(1 << 64)
