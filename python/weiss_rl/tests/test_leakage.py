from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.eval import HiddenInfoLeakagePair, build_hidden_info_leakage_diagnostics, write_leakage_diagnostics_json


def _pair(
    *,
    pair_id: int | str,
    public_a: list[int],
    public_b: list[int] | None = None,
    logits_a: list[float],
    logits_b: list[float],
    legal_mask_a: list[int],
    legal_mask_b: list[int] | None = None,
) -> HiddenInfoLeakagePair:
    return HiddenInfoLeakagePair(
        pair_id=pair_id,
        public_observation_a=public_a,
        public_observation_b=public_a if public_b is None else public_b,
        logits_a=logits_a,
        logits_b=logits_b,
        legal_mask_a=legal_mask_a,
        legal_mask_b=legal_mask_a if legal_mask_b is None else legal_mask_b,
    )


def test_build_hidden_info_leakage_diagnostics_rejects_public_parity_mismatch() -> None:
    with pytest.raises(ValueError, match="public parity violation"):
        build_hidden_info_leakage_diagnostics(
            [
                _pair(
                    pair_id="mismatch",
                    public_a=[1, 2, 3],
                    public_b=[1, 2, 4],
                    logits_a=[2.0, 0.0, -1.0],
                    logits_b=[2.0, 0.0, -1.0],
                    legal_mask_a=[1, 1, 0],
                )
            ]
        )


def test_build_hidden_info_leakage_diagnostics_applies_kl_and_tv_thresholds() -> None:
    payload = build_hidden_info_leakage_diagnostics(
        [
            _pair(
                pair_id="clean",
                public_a=[1, 0],
                logits_a=[3.0, 0.0, -1.0],
                logits_b=[3.0, 0.0, -1.0],
                legal_mask_a=[1, 1, 0],
            ),
            _pair(
                pair_id="mild",
                public_a=[2, 0],
                logits_a=[2.0, 0.5, -1.0],
                logits_b=[1.7, 0.8, -1.0],
                legal_mask_a=[1, 1, 0],
            ),
            _pair(
                pair_id="severe",
                public_a=[3, 0],
                logits_a=[3.0, 0.0, -1.0],
                logits_b=[0.0, 3.0, -1.0],
                legal_mask_a=[1, 1, 0],
            ),
        ],
        kl_median_threshold=0.01,
        kl_p95_threshold=0.05,
        tv_median_threshold=0.05,
        tv_p95_threshold=0.1,
    )

    assert payload["summary"]["thresholds_passed"] is False
    assert payload["summary"]["threshold_failures"] == ["kl_median", "kl_p95", "tv_median", "tv_p95"]
    assert payload["summary"]["kl_median"] > payload["thresholds"]["kl_median_threshold"]
    assert payload["summary"]["kl_p95"] > payload["thresholds"]["kl_p95_threshold"]
    assert payload["summary"]["tv_median"] > payload["thresholds"]["tv_median_threshold"]
    assert payload["summary"]["tv_p95"] > payload["thresholds"]["tv_p95_threshold"]
    assert payload["pairs"][0]["pair_id"] == "clean"
    assert payload["pairs"][2]["total_variation"] > 0.9


def test_write_leakage_diagnostics_json_persists_artifact(tmp_path: Path) -> None:
    payload = build_hidden_info_leakage_diagnostics(
        [
            _pair(
                pair_id="artifact",
                public_a=[7, 7],
                logits_a=[1.0, 0.0, -1.0],
                logits_b=[1.0, 0.0, -1.0],
                legal_mask_a=[1, 1, 0],
            )
        ]
    )

    path = tmp_path / "leakage.json"
    write_leakage_diagnostics_json(path, payload)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"]["thresholds_passed"] is True
    assert saved["pairs"][0]["pair_id"] == "artifact"
