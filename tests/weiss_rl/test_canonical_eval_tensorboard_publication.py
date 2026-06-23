from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.tensorboard_publication import (
    begin_canonical_eval_tensorboard_logging,
    publish_canonical_eval_tensorboard_summaries,
)

from .canonical_eval_publication_test_support import (
    FakeTensorBoardLogger,
    canonical_publication_layout,
    canonical_run_state,
    canonical_supplemental_outputs,
)


def test_canonical_tensorboard_publication_handles_enabled_and_disabled(tmp_path: Path, capsys) -> None:
    layout = canonical_publication_layout(tmp_path)
    enabled_logger = FakeTensorBoardLogger(enabled=True)
    run_state = canonical_run_state(layout=layout, tensorboard_logger=enabled_logger)
    supplemental = canonical_supplemental_outputs(tmp_path)

    begin_canonical_eval_tensorboard_logging(
        run_state=run_state,
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=enabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert enabled_logger.calls == [
        ("text", ("eval/run/manifest", {"run_id256": "ab" * 32})),
        ("final", ({"summary": "payload"}, 0)),
        ("metagame", ({"meta": "payload"}, layout.metagame_dir, 0)),
        ("readiness", ({"passed": True}, 0)),
    ]

    disabled_logger = FakeTensorBoardLogger(enabled=False)
    begin_canonical_eval_tensorboard_logging(
        run_state=canonical_run_state(
            layout=layout,
            tensorboard_logger=disabled_logger,
            manifest={},
            run_id256="",
        ),
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=disabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert disabled_logger.calls == []
    assert "TensorBoard logging is disabled for eval: SummaryWriter unavailable" in capsys.readouterr().err
