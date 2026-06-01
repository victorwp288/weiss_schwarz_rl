from __future__ import annotations

from weiss_rl.training.algorithm_families import STRUCTURED_VTRACE_ALGORITHMS


def validate_algorithm_model_contract(*, algorithm: str, recurrent_core: str, encoder_kind: str) -> None:
    normalized_core = str(recurrent_core).strip().lower()
    normalized_encoder = str(encoder_kind).strip().lower()
    if algorithm == "impala_vtrace_gru" and normalized_core != "gru":
        raise RuntimeError("impala_vtrace_gru requires model.recurrent_core=gru")
    if algorithm == "impala_vtrace_ff" and normalized_core != "none":
        raise RuntimeError("impala_vtrace_ff requires model.recurrent_core=none")
    if algorithm in STRUCTURED_VTRACE_ALGORITHMS and normalized_core not in {"gru", "none"}:
        raise RuntimeError(f"{algorithm} requires a supported model.recurrent_core value")
    if algorithm in STRUCTURED_VTRACE_ALGORITHMS and normalized_encoder != "structured_v2":
        raise RuntimeError(f"{algorithm} requires model.encoder_kind=structured_v2")
