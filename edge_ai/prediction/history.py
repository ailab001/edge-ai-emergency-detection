from __future__ import annotations

from collections import deque

DAY = 86400
DEFAULT_WINDOW_DAYS = 7
DEFAULT_MAX_SAMPLES = 10_080


class HistoryBuffer:
    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW_DAYS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ):
        self._window_sec = window_days * DAY
        self._buffers: dict[str, deque[tuple[float, float]]] = {}
        self._max_samples = max_samples

    def append(self, sensor_id: str, ts: float, value: float) -> None:
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self._max_samples)
        self._buffers[sensor_id].append((ts, value))

    def evict(self, now: float) -> None:
        cutoff = now - self._window_sec
        for buf in self._buffers.values():
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def get(self, sensor_id: str) -> list[tuple[float, float]]:
        return list(self._buffers.get(sensor_id, []))

    def get_values(self, sensor_id: str) -> list[float]:
        return [v for _, v in self._buffers.get(sensor_id, [])]

    def sensor_ids(self) -> list[str]:
        return list(self._buffers.keys())

    @property
    def window_days(self) -> int:
        return self._window_sec // DAY
