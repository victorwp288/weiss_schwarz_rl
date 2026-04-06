"""League self-play helpers."""

from .opponent_pool import OpponentPoolSampler, sample_opponent_snapshot_ids, select_opponent_snapshot_ids
from .pfsp import pfsp_probabilities
from .promotion_gate import (
    PromotionGateAnchor,
    PromotionGateAnchorResult,
    PromotionGatePosterior,
    PromotionGateRate,
    PromotionGateResult,
    resolve_promotion_gate_anchors,
    resolve_promotion_gate_seed_file,
    run_promotion_gate,
)
from .registry import SnapshotRegistry

__all__ = [
    "OpponentPoolSampler",
    "PromotionGateAnchor",
    "PromotionGateAnchorResult",
    "PromotionGatePosterior",
    "PromotionGateRate",
    "PromotionGateResult",
    "SnapshotRegistry",
    "pfsp_probabilities",
    "resolve_promotion_gate_anchors",
    "resolve_promotion_gate_seed_file",
    "run_promotion_gate",
    "sample_opponent_snapshot_ids",
    "select_opponent_snapshot_ids",
]
