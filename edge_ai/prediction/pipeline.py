from __future__ import annotations

import math

from shared.schemas import (
    EdgeCompactPayload,
    EdgePreprocessedData,
    ExpertEvent,
    HomeLocation,
    MobileGPS,
    SensorType,
    TransmissionTrigger,
)
from edge_ai.prediction.anomaly import AnomalyAggregator
from edge_ai.prediction.builder import GradePredictor
from edge_ai.prediction.gps import GPSAnalyzer
from edge_ai.prediction.history import HistoryBuffer
from edge_ai.prediction.predictor import FFTPredictor, TimeSeriesPredictor
from edge_ai.prediction.trend import TrendAnalyzer

PREDICTION_HORIZON = 60
MIN_SAMPLES_FOR_PREDICTION = 30


class EdgePredictionPipeline:
    def __init__(
        self,
        home_location: HomeLocation | None = None,
        predictor: TimeSeriesPredictor | None = None,
        window_days: int = 7,
    ):
        self._history = HistoryBuffer(window_days=window_days)
        self._trend = TrendAnalyzer()
        self._predictor = predictor or FFTPredictor()
        self._anomaly = AnomalyAggregator()
        self._grade = GradePredictor()
        self._gps = GPSAnalyzer(home_location) if home_location else None
        self._sensor_types: dict[str, SensorType] = {}

    def record(self, preprocessed: EdgePreprocessedData) -> None:
        for ch in preprocessed.channels:
            self._history.append(ch.sensor_id, ch.ts, ch.ewma_value)
            self._sensor_types[ch.sensor_id] = ch.sensor_type
        self._history.evict(preprocessed.cycle_ts)

    def update_gps(self, gps: MobileGPS) -> None:
        if self._gps:
            self._gps.update(gps)

    def build_payload(
        self,
        preprocessed: EdgePreprocessedData,
        trigger: TransmissionTrigger,
        expert_events: list[ExpertEvent] | None = None,
    ) -> EdgeCompactPayload:
        trend_summary = self._trend.analyze(self._history, self._sensor_types)
        anomaly_summary = self._anomaly.aggregate(preprocessed.channels)
        gps_summary = self._gps.evaluate(preprocessed.cycle_ts) if self._gps else None

        predicted_deviation = self._run_predictor()

        prediction = self._grade.predict(
            current_grade=preprocessed.situation_grade,
            trend=trend_summary,
            anomaly=anomaly_summary,
            scenarios=preprocessed.cross_scenarios,
            gps=gps_summary,
            predicted_deviation=predicted_deviation,
        )

        window_start = self._window_start()

        return EdgeCompactPayload(
            subject_id=preprocessed.subject_id,
            gateway_id=preprocessed.gateway_id,
            window_start=window_start,
            window_end=preprocessed.cycle_ts,
            transmission_trigger=trigger,
            edge_grade=preprocessed.situation_grade,
            trend_summary=trend_summary,
            anomaly_summary=anomaly_summary,
            prediction=prediction,
            gps_summary=gps_summary,
            expert_events=expert_events or [],
            sent_ts=preprocessed.cycle_ts,
        )

    def _run_predictor(self) -> float:
        deviations: list[float] = []

        for sid in self._history.sensor_ids():
            values = self._history.get_values(sid)
            if len(values) < MIN_SAMPLES_FOR_PREDICTION:
                continue

            predicted = self._predictor.predict(values, PREDICTION_HORIZON)
            if not predicted:
                continue

            n = len(values)
            mean = sum(values) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
            if std < 1e-9:
                continue

            pred_mean = sum(predicted) / len(predicted)
            deviations.append(abs(pred_mean - mean) / std)

        if not deviations:
            return 0.0
        return min(max(deviations) * 0.1, 0.2)

    def _window_start(self) -> float:
        earliest = float("inf")
        for sid in self._history.sensor_ids():
            pairs = self._history.get(sid)
            if pairs:
                earliest = min(earliest, pairs[0][0])
        return earliest if earliest < float("inf") else 0.0
