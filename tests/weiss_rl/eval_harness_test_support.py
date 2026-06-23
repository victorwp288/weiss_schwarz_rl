from __future__ import annotations

from weiss_rl.eval.harness import GameResult, ScheduledGame

_RUN_ID256 = "ab" * 32
_CONFIG_HASH256 = "cd" * 32
_SPEC_HASH256 = "ef" * 32


class _FakeRunner:
    def __init__(self, results: list[GameResult]) -> None:
        self._results = list(results)
        self.calls: list[ScheduledGame] = []

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        self.calls.append(scheduled_game)
        if not self._results:
            raise AssertionError("fake runner exhausted")
        return self._results.pop(0)
