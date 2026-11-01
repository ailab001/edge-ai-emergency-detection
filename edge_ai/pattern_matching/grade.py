from __future__ import annotations

from shared.schemas import CrossScenarioResult, ProcessedChannel, SituationGrade
from edge_ai.pattern_matching.thresholds import ThresholdManager


class GradeEvaluator:
    def evaluate(
        self,
        channels: list[ProcessedChannel],
        cross_results: list[CrossScenarioResult],
        thresholds: ThresholdManager,
    ) -> tuple[SituationGrade, str]:
        danger_triggered = [r for r in cross_results if r.triggered and r.confidence >= 0.7]
        if danger_triggered:
            top = max(danger_triggered, key=lambda r: r.confidence)
            return SituationGrade.DANGER, f"scenario:{top.scenario_type}"

        caution_triggered = [r for r in cross_results if r.triggered]
        if caution_triggered:
            top = max(caution_triggered, key=lambda r: r.confidence)
            return SituationGrade.CAUTION, f"scenario:{top.scenario_type}"

        danger_threshold = thresholds.get("grade.danger_score")
        caution_threshold = thresholds.get("grade.caution_score")

        anomaly_scores = [ch.anomaly_score for ch in channels if ch.is_anomaly]
        if anomaly_scores:
            max_score = max(anomaly_scores)
            if max_score >= danger_threshold:
                return SituationGrade.DANGER, f"anomaly_score:{max_score:.2f}"
            if max_score >= caution_threshold:
                return SituationGrade.CAUTION, f"anomaly_score:{max_score:.2f}"

        return SituationGrade.WATCH, "normal"
