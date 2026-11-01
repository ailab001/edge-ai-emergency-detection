from __future__ import annotations

from typing import Protocol

from shared.schemas import (
    CrossScenarioResult,
    ProcessedChannel,
    SensorType,
)
from edge_ai.pattern_matching.thresholds import ThresholdManager


class ScenarioChecker(Protocol):
    def check(
        self,
        channels: list[ProcessedChannel],
        thresholds: ThresholdManager,
        last_activity_ts: float | None,
        cycle_ts: float,
        baselines: dict[str, float] | None = None,
    ) -> CrossScenarioResult | None: ...


class HeatstrokeChecker:
    def check(
        self,
        channels: list[ProcessedChannel],
        thresholds: ThresholdManager,
        last_activity_ts: float | None,
        cycle_ts: float,
        baselines: dict[str, float] | None = None,
    ) -> CrossScenarioResult | None:
        temp_ch = _latest_by_type(channels, SensorType.TEMP)
        if temp_ch is None:
            return None

        temp_threshold = thresholds.get("heatstroke.temp_high")
        inactivity_limit = thresholds.get("heatstroke.inactivity_minutes") * 60

        high_temp = temp_ch.ewma_value >= temp_threshold

        inactive_duration = 0.0
        if last_activity_ts is not None:
            inactive_duration = cycle_ts - last_activity_ts

        long_inactive = inactive_duration >= inactivity_limit
        triggered = high_temp and long_inactive

        contributing = []
        if high_temp:
            contributing.append(temp_ch.sensor_id)
        if long_inactive:
            activity_ch = _latest_by_type(channels, SensorType.ACTIVITY)
            if activity_ch:
                contributing.append(activity_ch.sensor_id)

        confidence = 0.0
        if triggered:
            temp_excess = max(0.0, temp_ch.ewma_value - temp_threshold)
            confidence = min(1.0, 0.5 + temp_excess * 0.05 + inactive_duration / (inactivity_limit * 3))

        return CrossScenarioResult(
            scenario_type="heatstroke",
            triggered=triggered,
            confidence=confidence,
            contributing_sensors=contributing,
        )


class FallChecker:
    def check(
        self,
        channels: list[ProcessedChannel],
        thresholds: ThresholdManager,
        last_activity_ts: float | None,
        cycle_ts: float,
        baselines: dict[str, float] | None = None,
    ) -> CrossScenarioResult | None:
        pir_ch = _latest_by_type(channels, SensorType.PIR)
        activity_ch = _latest_by_type(channels, SensorType.ACTIVITY)
        if pir_ch is None and activity_ch is None:
            return None

        inactivity_limit = thresholds.get("fall.inactivity_minutes") * 60

        inactive_duration = 0.0
        if last_activity_ts is not None:
            inactive_duration = cycle_ts - last_activity_ts

        pir_silent = pir_ch is not None and pir_ch.ewma_value < 0.1
        long_inactive = inactive_duration >= inactivity_limit
        triggered = long_inactive and (pir_silent or pir_ch is None)

        contributing = []
        if activity_ch:
            contributing.append(activity_ch.sensor_id)
        if pir_ch and pir_silent:
            contributing.append(pir_ch.sensor_id)

        confidence = 0.0
        if triggered:
            confidence = min(1.0, 0.4 + inactive_duration / (inactivity_limit * 4))

        return CrossScenarioResult(
            scenario_type="fall",
            triggered=triggered,
            confidence=confidence,
            contributing_sensors=contributing,
        )


class HypothermiaChecker:
    def check(
        self,
        channels: list[ProcessedChannel],
        thresholds: ThresholdManager,
        last_activity_ts: float | None,
        cycle_ts: float,
        baselines: dict[str, float] | None = None,
    ) -> CrossScenarioResult | None:
        temp_ch = _latest_by_type(channels, SensorType.TEMP)
        activity_ch = _latest_by_type(channels, SensorType.ACTIVITY)
        if temp_ch is None:
            return None

        temp_threshold = thresholds.get("hypothermia.temp_low")
        low_temp = temp_ch.ewma_value <= temp_threshold

        activity_drop = False
        if activity_ch is not None and baselines:
            baseline = baselines.get(activity_ch.sensor_id)
            if baseline is not None and baseline > 0:
                drop_ratio = thresholds.get("hypothermia.activity_drop_ratio")
                activity_drop = activity_ch.ewma_value <= baseline * drop_ratio

        triggered = low_temp and activity_drop

        contributing = []
        if low_temp:
            contributing.append(temp_ch.sensor_id)
        if activity_drop and activity_ch:
            contributing.append(activity_ch.sensor_id)

        confidence = 0.0
        if triggered:
            temp_deficit = max(0.0, temp_threshold - temp_ch.ewma_value)
            confidence = min(1.0, 0.5 + temp_deficit * 0.1)

        return CrossScenarioResult(
            scenario_type="hypothermia",
            triggered=triggered,
            confidence=confidence,
            contributing_sensors=contributing,
        )


def _latest_by_type(
    channels: list[ProcessedChannel], sensor_type: SensorType
) -> ProcessedChannel | None:
    matches = [ch for ch in channels if ch.sensor_type == sensor_type]
    return max(matches, key=lambda c: c.ts) if matches else None
