from __future__ import annotations

from collections import deque

from shared.schemas import ProcessedChannel, SensorType
from edge_ai.pattern_matching.filters import EWMAFilter, KalmanFilter, SignalFilter

_DEFAULT_BASELINE_WINDOW = 60


class ChannelProcessor:
    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        filters: list[SignalFilter] | None = None,
        baseline_window: int = _DEFAULT_BASELINE_WINDOW,
    ):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self._filters: list[SignalFilter] = filters or [
            KalmanFilter(),
            EWMAFilter(),
        ]
        self._baseline_buf: deque[float] = deque(maxlen=baseline_window)
        self._baseline_sum: float = 0.0

    @property
    def baseline_mean(self) -> float | None:
        if not self._baseline_buf:
            return None
        return self._baseline_sum / len(self._baseline_buf)

    def process(self, raw_value: float, ts: float) -> ProcessedChannel:
        intermediate = raw_value
        stage_values: list[float] = []

        for f in self._filters:
            intermediate = f.update(intermediate, ts)
            stage_values.append(intermediate)

        kalman_value = stage_values[0] if len(stage_values) > 0 else raw_value
        ewma_value = stage_values[1] if len(stage_values) > 1 else kalman_value

        if len(self._baseline_buf) == self._baseline_buf.maxlen:
            self._baseline_sum -= self._baseline_buf[0]
        self._baseline_buf.append(ewma_value)
        self._baseline_sum += ewma_value

        return ProcessedChannel(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            raw_value=raw_value,
            kalman_value=kalman_value,
            ewma_value=ewma_value,
            anomaly_score=0.0,
            is_anomaly=False,
            ts=ts,
        )

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
        self._baseline_buf.clear()
        self._baseline_sum = 0.0
