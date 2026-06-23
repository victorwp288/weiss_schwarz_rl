from __future__ import annotations

import pytest
from weiss_rl.training.algorithm_contracts import validate_algorithm_model_contract


@pytest.mark.parametrize(
    ("algorithm", "recurrent_core", "encoder_kind"),
    [
        ("impala_vtrace_gru", "gru", "mlp"),
        ("impala_vtrace_ff", "none", "mlp"),
        ("structured_v2", "gru", "structured_v2"),
        ("structured_v2", "none", "structured_v2"),
        ("impala_vtrace_structured_v1", "gru", "structured_v2"),
    ],
)
def test_validate_algorithm_model_contract_accepts_supported_combinations(
    algorithm: str,
    recurrent_core: str,
    encoder_kind: str,
) -> None:
    validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=recurrent_core,
        encoder_kind=encoder_kind,
    )


@pytest.mark.parametrize(
    ("algorithm", "recurrent_core", "encoder_kind", "match"),
    [
        ("impala_vtrace_gru", "none", "mlp", "impala_vtrace_gru requires model.recurrent_core=gru"),
        ("impala_vtrace_ff", "gru", "mlp", "impala_vtrace_ff requires model.recurrent_core=none"),
        ("structured_v2", "lstm", "structured_v2", "structured_v2 requires a supported model.recurrent_core value"),
        (
            "impala_vtrace_structured_v1",
            "gru",
            "mlp",
            "impala_vtrace_structured_v1 requires model.encoder_kind=structured_v2",
        ),
    ],
)
def test_validate_algorithm_model_contract_rejects_unsupported_combinations(
    algorithm: str,
    recurrent_core: str,
    encoder_kind: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_algorithm_model_contract(
            algorithm=algorithm,
            recurrent_core=recurrent_core,
            encoder_kind=encoder_kind,
        )
