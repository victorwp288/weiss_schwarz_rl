from __future__ import annotations


class RecordingScope:
    def __init__(self, events: list[tuple[object, ...]], name: str) -> None:
        self._events = events
        self.name = name

    def __enter__(self) -> None:
        self._events.append(("enter", self.name))

    def __exit__(self, *_exc: object) -> None:
        self._events.append(("exit", self.name))


class SnapshotPublishingRuntime:
    def __init__(self, events: list[tuple[object, ...]], metrics: dict[str, float]) -> None:
        self._events = events
        self._metrics = metrics

    def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
        self._events.append(("snapshot", kwargs))
        return self._metrics
