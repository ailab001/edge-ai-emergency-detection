from __future__ import annotations

from shared.schemas import (
    ContextAdjustmentRecord,
    ContextSeverity,
    ExternalContext,
    PersonalProfile,
    SensorType,
    ThresholdAdjustment,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "heatstroke.temp_high": 35.0,
    "heatstroke.inactivity_minutes": 30.0,
    "fall.inactivity_minutes": 30.0,
    "hypothermia.temp_low": 15.0,
    "hypothermia.activity_drop_ratio": 0.3,
    "grade.caution_score": 0.3,
    "grade.danger_score": 0.6,
}

SEVERITY_SCALE: dict[ContextSeverity, float] = {
    ContextSeverity.ADVISORY: 0.0,
    ContextSeverity.WATCH: 0.1,
    ContextSeverity.WARNING: 0.2,
    ContextSeverity.EMERGENCY: 0.3,
}


class ThresholdManager:
    def __init__(self, base: dict[str, float] | None = None):
        self._base = dict(base or DEFAULT_THRESHOLDS)
        self._current = dict(self._base)
        self._adjustment_records: list[ContextAdjustmentRecord] = []

    def get(self, key: str) -> float:
        return self._current.get(key, 0.0)

    def apply_context(
        self, contexts: list[ExternalContext], now: float
    ) -> list[ContextAdjustmentRecord]:
        records: list[ContextAdjustmentRecord] = []
        self._current = dict(self._base)

        active = [c for c in contexts if c.effective_from <= now <= c.effective_until]

        for ctx in active:
            adjustments: list[ThresholdAdjustment] = []

            for adj in ctx.threshold_adjustments:
                key = f"{adj.target_scenario}.{adj.parameter_name}"
                if key in self._current:
                    self._current[key] = adj.adjusted_value
                    adjustments.append(adj)

            if self._is_heat_alert(ctx):
                adjustments.extend(self._lower_heatstroke_thresholds(ctx))

            if adjustments:
                records.append(ContextAdjustmentRecord(
                    context_id=ctx.context_id,
                    adjustments=adjustments,
                    applied_ts=now,
                ))

        self._adjustment_records = records
        return records

    def apply_profile(self, profile: PersonalProfile) -> None:
        if profile.mobility_impaired:
            key = "hypothermia.temp_low"
            self._current[key] = self._current.get(key, 15.0) + 3.0

        if profile.custom_thresholds:
            for k, v in profile.custom_thresholds.items():
                if k in self._current:
                    self._current[k] = v

    def get_adjustment_records(self) -> list[ContextAdjustmentRecord]:
        return list(self._adjustment_records)

    def _is_heat_alert(self, ctx: ExternalContext) -> bool:
        return (
            ctx.context_type == "weather_alert"
            and ctx.parameters.get("type") == "heat"
        )

    def _lower_heatstroke_thresholds(
        self, ctx: ExternalContext
    ) -> list[ThresholdAdjustment]:
        scale = SEVERITY_SCALE.get(ctx.severity, 0.0)
        if scale <= 0:
            return []

        adjustments: list[ThresholdAdjustment] = []

        key = "heatstroke.temp_high"
        original = self._base.get(key, 35.0)
        adjusted = original - (original * scale)
        self._current[key] = adjusted
        adjustments.append(ThresholdAdjustment(
            target_sensor_type=SensorType.TEMP,
            target_scenario="heatstroke",
            parameter_name="temp_high",
            original_value=original,
            adjusted_value=adjusted,
            reason=f"heat_alert_{ctx.severity.value}",
        ))

        key = "heatstroke.inactivity_minutes"
        original = self._base.get(key, 30.0)
        adjusted = original * (1.0 - scale)
        self._current[key] = adjusted
        adjustments.append(ThresholdAdjustment(
            target_sensor_type=SensorType.ACTIVITY,
            target_scenario="heatstroke",
            parameter_name="inactivity_minutes",
            original_value=original,
            adjusted_value=adjusted,
            reason=f"heat_alert_{ctx.severity.value}",
        ))

        return adjustments
