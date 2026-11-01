from __future__ import annotations

from shared.schemas import (
    AnomalySummary,
    CrossScenarioResult,
    EdgePrediction,
    GPSSummary,
    SituationGrade,
    TrendSummary,
)


class GradePredictor:
    def __init__(self, horizon_minutes: int = 60):
        self._horizon = horizon_minutes

    def predict(
        self,
        current_grade: SituationGrade,
        trend: TrendSummary,
        anomaly: AnomalySummary,
        scenarios: list[CrossScenarioResult],
        gps: GPSSummary | None = None,
        predicted_deviation: float = 0.0,
    ) -> EdgePrediction:
        risk = 0.0

        risk += max(0.0, anomaly.composite_score) * 0.3

        triggered = [s for s in scenarios if s.triggered]
        if triggered:
            max_conf = max(s.confidence for s in triggered)
            risk += max_conf * 0.25

        negative_slopes = [
            abs(ch.slope) for ch in trend.channels if ch.slope < -0.01
        ]
        if negative_slopes:
            risk += min(sum(negative_slopes) * 10, 0.15)

        if gps and gps.is_no_return_risk:
            risk += 0.1

        risk += min(max(0.0, predicted_deviation), 0.2)

        risk = min(1.0, max(0.0, risk))

        if risk >= 0.6:
            grade = SituationGrade.DANGER
        elif risk >= 0.3:
            grade = SituationGrade.CAUTION
        else:
            grade = SituationGrade.WATCH

        return EdgePrediction(
            predicted_grade=grade,
            predicted_risk_score=round(risk, 4),
            horizon_minutes=self._horizon,
        )
