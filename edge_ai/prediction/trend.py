from __future__ import annotations

import math

from shared.schemas import ChannelTrend, SensorType, TrendSummary
from edge_ai.prediction.history import HistoryBuffer


class TrendAnalyzer:
    def analyze(
        self, history: HistoryBuffer, sensor_types: dict[str, SensorType]
    ) -> TrendSummary:
        trends: list[ChannelTrend] = []

        for sid in history.sensor_ids():
            stype = sensor_types.get(sid)
            if stype is None:
                continue

            pairs = history.get(sid)
            if len(pairs) < 2:
                continue

            values = [v for _, v in pairs]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance)
            slope = self._slope_ts(pairs)

            trends.append(ChannelTrend(
                sensor_type=stype,
                mean=round(mean, 4),
                std=round(std, 4),
                slope=round(slope, 6),
                min=round(min(values), 4),
                max=round(max(values), 4),
            ))

        return TrendSummary(
            window_days=history.window_days,
            channels=trends,
        )

    @staticmethod
    def _slope_ts(pairs: list[tuple[float, float]]) -> float:
        n = len(pairs)
        if n < 2:
            return 0.0
        t0 = pairs[0][0]
        sum_x = sum(t - t0 for t, _ in pairs)
        sum_y = sum(v for _, v in pairs)
        sum_xx = sum((t - t0) ** 2 for t, _ in pairs)
        sum_xy = sum((t - t0) * v for t, v in pairs)
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom
