from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearningProgressDiagnosticSection:
    key: str
    question: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "question": self.question,
            "evidence": list(self.evidence),
        }


LEARNING_PROGRESS_DIAGNOSTIC_SECTIONS: tuple[LearningProgressDiagnosticSection, ...] = (
    LearningProgressDiagnosticSection(
        key="loss",
        question="Is the optimizer reducing the learner objective?",
        evidence=("training_metrics.jsonl loss windows",),
    ),
    LearningProgressDiagnosticSection(
        key="teacher_guidance",
        question="Is teacher supervision active and supported by collected rows?",
        evidence=("teacher accuracy", "teacher loss", "teacher support fractions"),
    ),
    LearningProgressDiagnosticSection(
        key="reward_scale",
        question="Are reward, advantage, and target magnitudes usable for learning?",
        evidence=("reward windows", "advantage windows", "target windows"),
    ),
    LearningProgressDiagnosticSection(
        key="chosen_action_learning",
        question="Are important chosen actions receiving trainable signal?",
        evidence=("chosen pass fractions", "mulligan fractions", "chosen-action advantage windows"),
    ),
    LearningProgressDiagnosticSection(
        key="action_distribution",
        question="Is the runtime producing plausible decisions instead of collapse patterns?",
        evidence=("collector action counters", "pass guards", "mulligan guards"),
    ),
    LearningProgressDiagnosticSection(
        key="off_policy",
        question="Are learner targets close enough to the actor behavior policy?",
        evidence=("V-trace rho", "clip rate", "target-behavior log-prob deltas"),
    ),
    LearningProgressDiagnosticSection(
        key="actor_model_sync",
        question="Are actors collecting with fresh enough learner weights?",
        evidence=("learner-actor lag", "policy-version lag"),
    ),
    LearningProgressDiagnosticSection(
        key="league_sampling",
        question="Is the league admitting and sampling useful opponents?",
        evidence=("PFSP pool size", "snapshot environment fractions", "quarantine counters"),
    ),
    LearningProgressDiagnosticSection(
        key="promotion_gate",
        question="Did checkpoints pass the gate that admits them into league play?",
        evidence=("promotion decisions", "anchor posterior means", "failure reasons"),
    ),
    LearningProgressDiagnosticSection(
        key="final_eval_matrix",
        question="Does final evaluation confirm the checkpoint selected during training?",
        evidence=("final-eval mean matrix", "best-row comparison", "checkpoint-best row"),
    ),
    LearningProgressDiagnosticSection(
        key="checkpoint_alias_integrity",
        question="Do checkpoint aliases point at their tracker source files?",
        evidence=("checkpoint tracker", "alias file hashes", "source checkpoint hashes"),
    ),
)


def learning_progress_diagnostic_plan_payload() -> list[dict[str, object]]:
    return [section.as_payload() for section in LEARNING_PROGRESS_DIAGNOSTIC_SECTIONS]


__all__ = [
    "LEARNING_PROGRESS_DIAGNOSTIC_SECTIONS",
    "LearningProgressDiagnosticSection",
    "learning_progress_diagnostic_plan_payload",
]
