from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

import weiss_rl.eval.harness as eval_harness
from weiss_rl.core.masking import masked_logp_from_legal_ids
from weiss_rl.eval import EvalSamplerAnomalies, sample_action_pinned, select_action_argmax_pinned
from weiss_rl.eval.harness import _normalize_cdf_probs
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


class _StubFloatRng:
    def __init__(self, *draws: float) -> None:
        self._draws = list(draws)
        self.calls = 0

    def next_float(self) -> float:
        if self.calls >= len(self._draws):
            raise AssertionError("stub rng exhausted")
        draw = self._draws[self.calls]
        self.calls += 1
        return draw


def _load_vectors() -> VectorPayload:
    payload = json.loads(TEST_VECTORS_PATH.read_text(encoding="utf-8"))
    return cast(VectorPayload, payload)


def _cases() -> list[VectorCase]:
    return _load_vectors()["cases"]


def _expected_single_row_logp(
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


def test_sample_action_pinned_matches_anchor_sequence() -> None:
    logits = np.array([0.0, -3.0, 0.5, -2.0, 1.0, 1.5, -4.0, -0.5], dtype=np.float32)
    legal_ids = np.array([0, 2, 4, 5, 7], dtype=np.uint32)
    rng = Pcg32XshRrV1(20260316)

    actions = [sample_action_pinned(logits, legal_ids, rng=rng)[0] for _ in range(10)]

    assert actions == PINNED_SAMPLER_ACTIONS


def test_sample_action_pinned_returns_masking_core_logp() -> None:
    logits = np.array([0.25, -9.0, -0.5, 8.0, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([0, 2, 4, 5], dtype=np.uint32)
    rng = _StubFloatRng(0.72)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 4
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_temperature_scales_model_distribution_and_logp() -> None:
    logits = np.array([0.0, 1.0, -10.0], dtype=np.float32)
    legal_ids = np.array([0, 1], dtype=np.uint32)
    rng = _StubFloatRng(0.2)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng, temperature=0.25)

    assert action == 1
    assert logp == pytest.approx(
        _expected_single_row_logp(logits, legal_ids, action, temperature=0.25),
        abs=1e-6,
    )
    assert rng.calls == 1


def test_select_action_argmax_pinned_uses_legal_argmax_without_rng() -> None:
    logits = np.array([2.0, 99.0, 0.5, 3.0, 3.0], dtype=np.float32)
    legal_ids = np.array([0, 2, 3, 4], dtype=np.uint32)

    action, logp = select_action_argmax_pinned(logits, legal_ids)

    assert action == 3
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)


def test_select_action_argmax_pinned_empty_legal_returns_pass() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)

    action, logp = select_action_argmax_pinned(logits, np.array([], dtype=np.uint32), pass_action_id=3)

    assert action == 3
    assert logp == pytest.approx(0.0)


def test_sample_action_pinned_rejects_non_finite_legal_logits() -> None:
    logits = np.array([0.25, np.nan, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([1, 2], dtype=np.uint32)
    rng = _StubFloatRng(0.25)

    with pytest.raises(ValueError, match="legal logits must be finite"):
        sample_action_pinned(logits, legal_ids, rng=rng)

    assert rng.calls == 0


def test_sample_action_pinned_ignores_non_finite_illegal_logits() -> None:
    logits = np.array([0.25, np.nan, 1.5, np.inf, 0.75], dtype=np.float32)
    legal_ids = np.array([0, 2, 4], dtype=np.uint32)
    rng = _StubFloatRng(0.6)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_empty_legal_returns_pass_without_rng_draw() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([], dtype=np.uint32)
    rng = _StubFloatRng(0.4)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng, pass_action_id=3)

    assert action == 3
    assert logp == pytest.approx(0.0)
    assert rng.calls == 0


def test_sample_action_pinned_singleton_legal_set_consumes_rng() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([2], dtype=np.uint32)
    rng = _StubFloatRng(0.999999)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(0.0)
    assert rng.calls == 1


def test_sample_action_pinned_skips_zero_prob_plateau_at_draw_zero() -> None:
    logits = np.array([-1000.0, 0.0, 0.0], dtype=np.float32)
    legal_ids = np.array([0, 1, 2], dtype=np.uint32)
    rng = _StubFloatRng(0.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 1
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_pins_final_bin_at_draw_one_with_zero_prob_tail() -> None:
    logits = np.array([0.0, -1000.0, -1000.0], dtype=np.float32)
    legal_ids = np.array([0, 1, 2], dtype=np.uint32)
    rng = _StubFloatRng(1.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_rounding_guard_pins_final_cdf_bin() -> None:
    logits = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    legal_ids = np.array([0, 1, 2, 3], dtype=np.uint32)
    rng = _StubFloatRng(1.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 3
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_normalize_cdf_probs_counts_renormalization_anomaly() -> None:
    probs64 = np.array([0.6, 0.400002], dtype=np.float64)
    anomalies = EvalSamplerAnomalies()

    normalized = _normalize_cdf_probs(probs64, anomalies=anomalies)

    assert float(np.sum(normalized, dtype=np.float64)) == pytest.approx(1.0)
    assert anomalies.cdf_renormalizations == 1


def test_sample_action_pinned_plumbs_renormalization_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = np.array([0.0, 0.5], dtype=np.float32)
    legal_ids = np.array([0, 1], dtype=np.uint32)
    anomalies = EvalSamplerAnomalies()

    def _fake_legal_probs_for_cdf(
        logits: np.ndarray,
        legal_ids: np.ndarray,
        *,
        anomalies: EvalSamplerAnomalies | None = None,
    ) -> np.ndarray:
        del logits, legal_ids
        return _normalize_cdf_probs(np.array([0.6, 0.400002], dtype=np.float64), anomalies=anomalies)

    monkeypatch.setattr(eval_harness, "_legal_probs_for_cdf", _fake_legal_probs_for_cdf)

    action, logp = sample_action_pinned(logits, legal_ids, rng=_StubFloatRng(0.75), anomalies=anomalies)

    assert action == 1
    assert logp == pytest.approx(_expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert anomalies.cdf_renormalizations == 1


def test_sample_action_pinned_empty_rows_do_not_consume_rng_but_non_empty_rows_do() -> None:
    logits = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    rng = _StubFloatRng(0.2)

    pass_action, pass_logp = sample_action_pinned(
        logits,
        np.array([], dtype=np.uint32),
        rng=rng,
        pass_action_id=1,
    )
    sampled_action, sampled_logp = sample_action_pinned(
        logits,
        np.array([0, 2, 3], dtype=np.uint32),
        rng=rng,
    )

    assert pass_action == 1
    assert pass_logp == pytest.approx(0.0)
    assert sampled_action == 2
    assert sampled_logp == pytest.approx(
        _expected_single_row_logp(logits, np.array([0, 2, 3], dtype=np.uint32), sampled_action),
        abs=1e-6,
    )
    assert rng.calls == 1
