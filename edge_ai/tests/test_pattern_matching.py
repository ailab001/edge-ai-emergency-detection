import pytest

from shared.schemas import (
    ContextSeverity,
    ProcessedChannel,
    SensorType,
    SituationGrade,
)
from edge_ai.pattern_matching import (
    KalmanFilter,
    EWMAFilter,
    ChannelProcessor,
    IsolationForestDetector,
    ThresholdManager,
    GradeEvaluator,
    HeatstrokeChecker,
    FallChecker,
    HypothermiaChecker,
    EdgePipeline,
)
from edge_ai.tests.conftest import (
    BASE_TS,
    make_temp,
    make_humidity,
    make_activity,
    make_pir,
    make_heat_alert,
    generate_normal_day,
    generate_heatstroke_scenario,
    generate_fall_scenario,
    generate_hypothermia_scenario,
)


class TestKalmanFilter:
    def test_first_value_passthrough(self):
        kf = KalmanFilter()
        assert kf.update(10.0, BASE_TS) == 10.0

    def test_smoothing(self):
        kf = KalmanFilter()
        kf.update(10.0, BASE_TS)
        smoothed = kf.update(20.0, BASE_TS + 1)
        assert 10.0 < smoothed < 20.0

    def test_reset(self):
        kf = KalmanFilter()
        kf.update(10.0, BASE_TS)
        kf.reset()
        assert kf.update(50.0, BASE_TS + 10) == 50.0


class TestEWMAFilter:
    def test_first_value_passthrough(self):
        ewma = EWMAFilter(span=10)
        assert ewma.update(10.0, BASE_TS) == 10.0

    def test_weighted_average(self):
        ewma = EWMAFilter(span=10)
        ewma.update(10.0, BASE_TS)
        result = ewma.update(20.0, BASE_TS + 1)
        assert 10.0 < result < 20.0
        assert result < 15.0


class TestChannelProcessor:
    def test_process_returns_processed_channel(self):
        cp = ChannelProcessor("T01", SensorType.TEMP)
        result = cp.process(25.0, BASE_TS)
        assert isinstance(result, ProcessedChannel)
        assert result.sensor_id == "T01"
        assert result.raw_value == 25.0

    def test_baseline_accumulates(self):
        cp = ChannelProcessor("T01", SensorType.TEMP, baseline_window=5)
        for i in range(5):
            cp.process(20.0 + i, BASE_TS + i * 60)
        assert cp.baseline_mean is not None
        assert 20.0 <= cp.baseline_mean <= 24.0


class TestThresholdManager:
    def test_default_values(self):
        tm = ThresholdManager()
        assert tm.get("heatstroke.temp_high") == 35.0
        assert tm.get("hypothermia.temp_low") == 15.0

    def test_heat_alert_lowers_threshold(self):
        tm = ThresholdManager()
        ctx = make_heat_alert(BASE_TS, ContextSeverity.WARNING)
        tm.apply_context([ctx], BASE_TS + 100)
        assert tm.get("heatstroke.temp_high") < 35.0

    def test_profile_mobility_raises_hypothermia(self, impaired_profile):
        tm = ThresholdManager()
        tm.apply_profile(impaired_profile)
        assert tm.get("hypothermia.temp_low") == 18.0


class TestScenarioCheckers:
    def _make_channels(self, temp: float, activity: float, pir: float, ts: float):
        return [
            ProcessedChannel(
                sensor_id="T01", sensor_type=SensorType.TEMP,
                raw_value=temp, kalman_value=temp, ewma_value=temp,
                anomaly_score=0.0, is_anomaly=False, ts=ts,
            ),
            ProcessedChannel(
                sensor_id="A01", sensor_type=SensorType.ACTIVITY,
                raw_value=activity, kalman_value=activity, ewma_value=activity,
                anomaly_score=0.0, is_anomaly=False, ts=ts,
            ),
            ProcessedChannel(
                sensor_id="P01", sensor_type=SensorType.PIR,
                raw_value=pir, kalman_value=pir, ewma_value=pir,
                anomaly_score=0.0, is_anomaly=False, ts=ts,
            ),
        ]

    def test_heatstroke_triggered(self):
        checker = HeatstrokeChecker()
        tm = ThresholdManager()
        channels = self._make_channels(36.0, 0, 0, BASE_TS)
        last_activity = BASE_TS - 2000
        result = checker.check(channels, tm, last_activity, BASE_TS)
        assert result is not None
        assert result.triggered is True
        assert result.scenario_type == "heatstroke"

    def test_heatstroke_not_triggered_normal_temp(self):
        checker = HeatstrokeChecker()
        tm = ThresholdManager()
        channels = self._make_channels(24.0, 0, 0, BASE_TS)
        result = checker.check(channels, tm, BASE_TS - 2000, BASE_TS)
        assert result is not None
        assert result.triggered is False

    def test_fall_triggered(self):
        checker = FallChecker()
        tm = ThresholdManager()
        channels = self._make_channels(24.0, 0, 0, BASE_TS)
        last_activity = BASE_TS - 2000
        result = checker.check(channels, tm, last_activity, BASE_TS)
        assert result is not None
        assert result.triggered is True

    def test_hypothermia_triggered(self):
        checker = HypothermiaChecker()
        tm = ThresholdManager()
        channels = self._make_channels(13.0, 0.1, 0, BASE_TS)
        baselines = {"A01": 5.0}
        result = checker.check(channels, tm, BASE_TS - 100, BASE_TS, baselines)
        assert result is not None
        assert result.triggered is True


class TestGradeEvaluator:
    def test_watch_on_normal(self):
        ge = GradeEvaluator()
        tm = ThresholdManager()
        grade, reason = ge.evaluate([], [], tm)
        assert grade == SituationGrade.WATCH

    def test_danger_on_high_confidence_scenario(self):
        from shared.schemas import CrossScenarioResult
        ge = GradeEvaluator()
        tm = ThresholdManager()
        scenarios = [CrossScenarioResult(
            scenario_type="heatstroke", triggered=True,
            confidence=0.8, contributing_sensors=["T01"],
        )]
        grade, _ = ge.evaluate([], scenarios, tm)
        assert grade == SituationGrade.DANGER


class TestEdgePipelineIntegration:
    def test_normal_day_stays_watch(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        readings = generate_normal_day(BASE_TS, duration_hours=2)

        for r in readings:
            pipeline.ingest(r)

        result = pipeline.run_cycle(BASE_TS + 7200)
        assert result.situation_grade == SituationGrade.WATCH
        assert result.subject_id == "U001"
        assert len(result.channels) > 0

    def test_heatstroke_scenario_escalates(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        readings = generate_heatstroke_scenario(BASE_TS)

        for r in readings:
            pipeline.ingest(r)

        result = pipeline.run_cycle(BASE_TS + 4200)
        heatstroke = [s for s in result.cross_scenarios if s.scenario_type == "heatstroke"]
        assert len(heatstroke) == 1
        assert heatstroke[0].triggered is True

    def test_fall_scenario_escalates(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        readings = generate_fall_scenario(BASE_TS)

        for r in readings:
            pipeline.ingest(r)

        result = pipeline.run_cycle(BASE_TS + 3300)
        fall = [s for s in result.cross_scenarios if s.scenario_type == "fall"]
        assert len(fall) == 1
        assert fall[0].triggered is True

    def test_output_schema_valid(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        readings = generate_normal_day(BASE_TS, duration_hours=1)
        for r in readings:
            pipeline.ingest(r)

        result = pipeline.run_cycle(BASE_TS + 3600)
        dumped = result.model_dump()
        assert "subject_id" in dumped
        assert "channels" in dumped
        assert "situation_grade" in dumped
        assert "cross_scenarios" in dumped

    def test_context_adjustment_applied(self, gateway_config, normal_profile):
        pipeline = EdgePipeline(gateway_config, normal_profile)
        ctx = make_heat_alert(BASE_TS, ContextSeverity.WARNING)
        pipeline.update_context(ctx)

        readings = generate_heatstroke_scenario(BASE_TS)
        for r in readings:
            pipeline.ingest(r)

        result = pipeline.run_cycle(BASE_TS + 4200)
        assert len(result.context_adjustments) > 0
