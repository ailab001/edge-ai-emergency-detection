import pytest

from shared.schemas import (
    ActionType,
    EdgeCompactPayload,
    EdgePrediction,
    ExpertEvent,
    SituationGrade,
    TransmissionTrigger,
    TrendSummary,
    ChannelTrend,
    AnomalySummary,
    AnomalyRank,
    SensorType,
)
from edge_ai.expert_system import ExpertEngine
from edge_ai.pattern_matching import EdgePipeline
from edge_ai.tests.conftest import (
    BASE_TS,
    make_fire,
    make_emergency,
    make_temp,
    make_activity,
    make_pir,
    generate_heatstroke_scenario,
    generate_fire_event,
    generate_fire_cancelled,
)


class TestExpertAndPipelineIndependence:
    def test_expert_ignores_numeric_sensors(self):
        engine = ExpertEngine(subject_id="U001")
        assert engine.handle_reading(make_temp(BASE_TS, 40.0)) is None
        assert engine.handle_reading(make_activity(BASE_TS, 1)) is None
        assert engine.handle_reading(make_pir(BASE_TS, 1)) is None

    def test_pipeline_ignores_event_sensors(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        pipeline.ingest(make_fire(BASE_TS, "smoke"))
        pipeline.ingest(make_emergency(BASE_TS, "red"))
        result = pipeline.run_cycle(BASE_TS + 60)
        assert len(result.channels) == 0


class TestExpertEventToCompactPayload:
    def test_expert_event_fits_compact_schema(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        events = engine.check_timeouts(BASE_TS + 16.0)
        assert len(events) == 1

        expert_event = events[0]
        payload = EdgeCompactPayload(
            subject_id="U001",
            gateway_id="gw_001",
            window_start=BASE_TS,
            window_end=BASE_TS + 16.0,
            transmission_trigger=TransmissionTrigger.EXPERT_EVENT,
            edge_grade=SituationGrade.DANGER,
            trend_summary=TrendSummary(window_days=7, channels=[]),
            anomaly_summary=AnomalySummary(composite_score=0.0, top_anomalies=[]),
            prediction=EdgePrediction(
                predicted_grade=SituationGrade.DANGER,
                predicted_risk_score=1.0,
                horizon_minutes=0,
            ),
            expert_events=[expert_event],
            sent_ts=BASE_TS + 16.0,
        )
        assert payload.expert_events[0].event_type == "fire"
        assert payload.expert_events[0].cancelled is False
        assert payload.transmission_trigger == TransmissionTrigger.EXPERT_EVENT


class TestPipelineOutputToCompactPayload:
    def test_preprocessed_converts_to_compact(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        readings = generate_heatstroke_scenario(BASE_TS)
        for r in readings:
            pipeline.ingest(r)

        preprocessed = pipeline.run_cycle(BASE_TS + 4200)

        top_anomalies = []
        for ch in preprocessed.channels:
            if ch.is_anomaly:
                top_anomalies.append(AnomalyRank(
                    sensor_type=ch.sensor_type,
                    score=ch.anomaly_score,
                    sensor_id=ch.sensor_id,
                ))

        composite = max(
            (ch.anomaly_score for ch in preprocessed.channels),
            default=0.0,
        )

        payload = EdgeCompactPayload(
            subject_id=preprocessed.subject_id,
            gateway_id=preprocessed.gateway_id,
            window_start=BASE_TS,
            window_end=BASE_TS + 4200,
            transmission_trigger=TransmissionTrigger.GRADE_ESCALATION,
            edge_grade=preprocessed.situation_grade,
            trend_summary=TrendSummary(
                window_days=7,
                channels=[ChannelTrend(
                    sensor_type=ch.sensor_type,
                    mean=ch.ewma_value,
                    std=0.0,
                    slope=0.0,
                    min=ch.ewma_value,
                    max=ch.ewma_value,
                ) for ch in preprocessed.channels],
            ),
            anomaly_summary=AnomalySummary(
                composite_score=max(-1.0, min(1.0, composite)),
                top_anomalies=top_anomalies[:3],
            ),
            prediction=EdgePrediction(
                predicted_grade=preprocessed.situation_grade,
                predicted_risk_score=0.5,
                horizon_minutes=60,
            ),
            sent_ts=BASE_TS + 4200,
        )
        assert payload.subject_id == "U001"
        assert payload.edge_grade in (
            SituationGrade.WATCH, SituationGrade.CAUTION, SituationGrade.DANGER,
        )


class TestJudgmentLogForCoreAI:
    def test_dispatch_log_structure(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.check_timeouts(BASE_TS + 16.0)

        logs = engine.get_judgment_logs()
        assert len(logs) == 1
        log = logs[0]
        assert log.subject_id == "U001"
        assert log.sensor_type == SensorType.FIRE
        assert log.resolution == "dispatched"
        assert log.action_taken == ActionType.DISPATCH_119
        assert log.cancel_window_sec == 15.0

    def test_cancelled_log_structure(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.handle_reading(make_fire(BASE_TS + 8.0, "normal"))

        logs = engine.get_judgment_logs()
        assert len(logs) == 1
        assert logs[0].resolution == "cancelled"
        assert logs[0].cancel_reading is not None

    def test_log_buffer_clears_after_read(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.check_timeouts(BASE_TS + 16.0)

        first = engine.get_judgment_logs()
        second = engine.get_judgment_logs()
        assert len(first) == 1
        assert len(second) == 0


class TestConcurrentExpertAndPipeline:
    def test_fire_during_heatstroke(self, gateway_config, normal_profile):
        expert = ExpertEngine(subject_id="U001")
        pipeline = EdgePipeline(gateway_config, normal_profile)

        readings = generate_heatstroke_scenario(BASE_TS)
        for r in readings:
            pipeline.ingest(r)
            expert_result = expert.handle_reading(r)
            assert expert_result is None

        fire_readings = generate_fire_event(BASE_TS + 3000)
        for r in fire_readings:
            expert.handle_reading(r)

        expert_events = expert.check_timeouts(BASE_TS + 3016)
        preprocessed = pipeline.run_cycle(BASE_TS + 4200)

        assert len(expert_events) == 1
        assert expert_events[0].action_taken == ActionType.DISPATCH_119

        heatstroke = [s for s in preprocessed.cross_scenarios if s.scenario_type == "heatstroke"]
        assert len(heatstroke) == 1
        assert heatstroke[0].triggered is True
