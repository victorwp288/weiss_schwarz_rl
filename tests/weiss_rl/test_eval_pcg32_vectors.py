from __future__ import annotations

import pytest
from weiss_rl.eval.sampling.rng_pcg32 import NEXT_U64_ORDER, PCG32_XSH_RR_V1, Pcg32XshRrV1

from .eval_sampler_test_support import (
    PINNED_VECTOR_SEEDS,
    SEEDED_STREAM_ANCHOR,
    VectorCase,
    load_vectors,
    vector_cases,
)


@pytest.mark.parametrize("case", vector_cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_u32_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [rng.next_u32() for _ in range(10)]
    assert outputs == case["uint32_outputs"]


@pytest.mark.parametrize("case", vector_cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_u64_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [f"0x{rng.next_u64():016x}" for _ in range(5)]
    assert outputs == case["uint64_outputs"]


@pytest.mark.parametrize("case", vector_cases(), ids=lambda case: f"seed={case['seed64']}")
def test_pcg32_next_float_matches_golden_vectors(case: VectorCase) -> None:
    rng = Pcg32XshRrV1(int(case["seed64"]))
    outputs = [rng.next_float().hex() for _ in range(5)]
    assert outputs == case["uniform01_hex"]


def test_pcg32_vector_corpus_shape_is_pinned() -> None:
    cases = vector_cases()
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
    payload = load_vectors()
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
