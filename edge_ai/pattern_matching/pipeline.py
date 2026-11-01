from __future__ import annotations

from shared.schemas import (
    ActivityReading,
    DoorReading,
    EdgePreprocessedData,
    ExternalContext,
    GatewayConfig,
    HumidityReading,
    IlluminanceReading,
    MotionReading,
    PersonalProfile,
    ProcessedChannel,
    SensorReading,
    SensorType,
    TemperatureReading,
)
from edge_ai.pattern_matching.channel import ChannelProcessor
from edge_ai.pattern_matching.detectors import AnomalyDetector, IsolationForestDetector
from edge_ai.pattern_matching.grade import GradeEvaluator
from edge_ai.pattern_matching.scenarios import (
    FallChecker,
    HeatstrokeChecker,
    HypothermiaChecker,
    ScenarioChecker,
)
from edge_ai.pattern_matching.thresholds import ThresholdManager

NUMERIC_SENSOR_TYPES = {
    SensorType.TEMP,
    SensorType.HUMI,
    SensorType.LUX,
    SensorType.ACTIVITY,
    SensorType.PIR,
}


class EdgePipeline:
    def __init__(
        self,
        gateway_config: GatewayConfig,
        profile: PersonalProfile | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        scenario_checkers: list[ScenarioChecker] | None = None,
        grade_evaluator: GradeEvaluator | None = None,
        threshold_manager: ThresholdManager | None = None,
    ):
        self._config = gateway_config
        self._profile = profile
        self._detector = anomaly_detector or IsolationForestDetector()
        self._checkers = scenario_checkers or [
            HeatstrokeChecker(),
            FallChecker(),
            HypothermiaChecker(),
        ]
        self._grade_evaluator = grade_evaluator or GradeEvaluator()
        self._thresholds = threshold_manager or ThresholdManager()

        if profile:
            self._thresholds.apply_profile(profile)

        self._channels: dict[str, ChannelProcessor] = {}
        self._latest: dict[str, ProcessedChannel] = {}
        self._last_activity_ts: float | None = None
        self._first_ingest_ts: float | None = None
        self._contexts: list[ExternalContext] = []

    def ingest(self, reading: SensorReading) -> None:
        value = self._extract_numeric(reading)
        if value is None:
            return

        if self._first_ingest_ts is None:
            self._first_ingest_ts = reading.ts

        sensor_id = reading.sensor_id
        if sensor_id not in self._channels:
            self._channels[sensor_id] = ChannelProcessor(
                sensor_id=sensor_id,
                sensor_type=reading.sensor_type,
            )

        processed = self._channels[sensor_id].process(value, reading.ts)
        self._latest[sensor_id] = processed

        if reading.sensor_type in (SensorType.ACTIVITY, SensorType.PIR):
            if value > 0:
                self._last_activity_ts = reading.ts

    def update_context(self, context: ExternalContext) -> None:
        self._contexts = [
            c for c in self._contexts if c.context_id != context.context_id
        ]
        self._contexts.append(context)

    def run_cycle(self, cycle_ts: float) -> EdgePreprocessedData:
        self._thresholds.apply_context(self._contexts, cycle_ts)
        if self._profile:
            self._thresholds.apply_profile(self._profile)

        channels = list(self._latest.values())
        channels = self._detector.fit_predict(channels)

        baselines = {
            sid: cp.baseline_mean
            for sid, cp in self._channels.items()
            if cp.baseline_mean is not None
        }

        cross_results = []
        inactivity_since = self._last_activity_ts or self._first_ingest_ts
        for checker in self._checkers:
            result = checker.check(
                channels, self._thresholds, inactivity_since, cycle_ts, baselines,
            )
            if result is not None:
                cross_results.append(result)

        grade, reason = self._grade_evaluator.evaluate(
            channels, cross_results, self._thresholds,
        )

        return EdgePreprocessedData(
            subject_id=self._config.subject_id,
            gateway_id=self._config.gateway_id,
            channels=channels,
            cross_scenarios=cross_results,
            situation_grade=grade,
            grade_reason=reason,
            cycle_ts=cycle_ts,
            context_adjustments=self._thresholds.get_adjustment_records(),
        )

    @staticmethod
    def _extract_numeric(
        reading: SensorReading,
    ) -> float | None:
        if isinstance(reading, (TemperatureReading, HumidityReading, IlluminanceReading)):
            return reading.value
        if isinstance(reading, (ActivityReading, MotionReading)):
            return float(reading.value)
        return None
