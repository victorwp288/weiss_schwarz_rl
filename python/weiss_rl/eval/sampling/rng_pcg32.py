"""Pinned PCG32 XSH RR RNG for paper-grade evaluation."""

from __future__ import annotations

import operator

from weiss_rl.artifacts.reproducibility import stable_hash64

PCG32_XSH_RR_V1 = "pcg32_xsh_rr_v1"
NEXT_U64_ORDER = "hi_then_lo"

_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1
_PCG32_MULTIPLIER = 6364136223846793005
_FLOAT53_DENOMINATOR = float(1 << 53)
_STATE_TAG = b"pcg32_state_v1"
_SEQ_TAG = b"pcg32_seq_v1"

__all__ = ["NEXT_U64_ORDER", "PCG32_XSH_RR_V1", "Pcg32XshRrV1"]


def _require_u64(value: int, name: str) -> int:
    number = operator.index(value)
    if not 0 <= number <= _U64_MASK:
        raise ValueError(f"{name} must be in [0, 2**64 - 1], got {number}")
    return number


def _u64_le(value: int) -> bytes:
    return _require_u64(value, "value").to_bytes(8, byteorder="little", signed=False)


def _rotate_right_u32(value: int, rot: int) -> int:
    rot &= 31
    return ((value >> rot) | (value << ((-rot) & 31))) & _U32_MASK


def _derive_stream(seed64: int) -> tuple[int, int]:
    seed_bytes = _u64_le(seed64)
    initstate = stable_hash64(_STATE_TAG + seed_bytes)
    initseq = stable_hash64(_SEQ_TAG + seed_bytes)
    return initstate, initseq


class Pcg32XshRrV1:
    """Versioned, seeded PCG32 XSH RR generator.

    Seeding is pinned to the master plan:
    - initstate = stable_hash64(b"pcg32_state_v1" + rng_seed64_le)
    - initseq = stable_hash64(b"pcg32_seq_v1" + rng_seed64_le)
    - recommended PCG two-step scramble

    next_u64() concatenates two uint32 draws as hi||lo.
    next_float() converts the top 53 bits with (r + 0.5) / 2**53.
    """

    __slots__ = ("_inc", "_state", "seed64")

    def __init__(self, rng_seed64: int) -> None:
        seed64 = _require_u64(rng_seed64, "rng_seed64")
        initstate, initseq = _derive_stream(seed64)

        self.seed64 = seed64
        self._state = 0
        self._inc = ((initseq << 1) | 1) & _U64_MASK

        self.next_u32()
        self._state = (self._state + initstate) & _U64_MASK
        self.next_u32()

    def next_u32(self) -> int:
        oldstate = self._state
        self._state = (oldstate * _PCG32_MULTIPLIER + self._inc) & _U64_MASK
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & _U32_MASK
        rot = (oldstate >> 59) & 31
        return _rotate_right_u32(xorshifted, rot)

    def next_u64(self) -> int:
        hi = self.next_u32()
        lo = self.next_u32()
        return ((hi << 32) | lo) & _U64_MASK

    def next_float(self) -> float:
        x = self.next_u64()
        r = x >> 11
        return (r + 0.5) / _FLOAT53_DENOMINATOR
