from __future__ import annotations

from shared.schemas import (
    AnomalyRank,
    AnomalySummary,
    ProcessedChannel,
    SensorType,
)

DEFAULT_WEIGHTS: dict[SensorType, float] = {
    SensorType.ACTIVITY: 1.5,
    SensorType.PIR: 1.3,
    SensorType.TEMP: 1.0,
    SensorType.HUMI: 0.6,
    SensorType.LUX: 0.5,
}


class AnomalyAggregator:
    def __init__(self, weights: dict[SensorType, float] | None = None):
        self._weights = weights or DEFAULT_WEIGHTS

    def aggregate(self, channels: list[ProcessedChannel]) -> AnomalySummary:
        if not channels:
            return AnomalySummary(composite_score=0.0, top_anomalies=[])

        weighted_sum = 0.0
        weight_total = 0.0

        ranked: list[AnomalyRank] = []
        for ch in channels:
            w = self._weights.get(ch.sensor_type, 1.0)
            weighted_sum += ch.anomaly_score * w
            weight_total += w

            ranked.append(AnomalyRank(
                sensor_type=ch.sensor_type,
                score=ch.anomaly_score,
                sensor_id=ch.sensor_id,
            ))

        composite = weighted_sum / weight_total if weight_total > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))

        ranked.sort(key=lambda r: abs(r.score), reverse=True)

        return AnomalySummary(
            composite_score=round(composite, 4),
            top_anomalies=ranked[:3],
        )
