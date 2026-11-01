import math

import pytest

from shared.schemas import (
    AnomalyRank,
    CrossScenarioResult,
    EdgeCompactPayload,
    EdgePreprocessedData,
    GPSSummary,
    HomeLocation,
    MobileGPS,
    ProcessedChannel,
    SensorType,
    SituationGrade,
    TransmissionTrigger,
)
from edge_ai.prediction import (
    AnomalyAggregator,
    EdgePredictionPipeline,
    FFTPredictor,
    GPSAnalyzer,
    GradePredictor,
    HistoryBuffer,
    TrendAnalyzer,
    haversine,
)
from edge_ai.tests.conftest import (
    BASE_TS,
    make_gps,
    generate_normal_day,
    generate_heatstroke_scenario,
)


class TestHistoryBuffer:
    def test_append_and_get(self):
        buf = HistoryBuffer(window_days=7)
        buf.append("T01", BASE_TS, 24.0)
        buf.append("T01", BASE_TS + 60, 24.5)
        assert len(buf.get("T01")) == 2
        assert buf.get_values("T01") == [24.0, 24.5]

    def test_eviction(self):
        buf = HistoryBuffer(window_days=1)
        buf.append("T01", BASE_TS, 24.0)
        buf.append("T01", BASE_TS + 90000, 25.0)
        buf.evict(BASE_TS + 90000)
        assert len(buf.get("T01")) == 1
        assert buf.get_values("T01") == [25.0]

    def test_sensor_ids(self):
        buf = HistoryBuffer()
        buf.append("T01", BASE_TS, 24.0)
        buf.append("H01", BASE_TS, 55.0)
        assert set(buf.sensor_ids()) == {"T01", "H01"}

    def test_empty_get(self):
        buf = HistoryBuffer()
        assert buf.get("nonexistent") == []


class TestTrendAnalyzer:
    def test_basic_trend(self):
        buf = HistoryBuffer()
        for i in range(100):
            buf.append("T01", BASE_TS + i * 60, 20.0 + i * 0.1)

        analyzer = TrendAnalyzer()
        summary = analyzer.analyze(buf, {"T01": SensorType.TEMP})
        assert len(summary.channels) == 1
        trend = summary.channels[0]
        assert trend.sensor_type == SensorType.TEMP
        assert trend.slope > 0
        assert trend.min < trend.max

    def test_flat_trend(self):
        buf = HistoryBuffer()
        for i in range(50):
            buf.append("T01", BASE_TS + i * 60, 24.0)

        analyzer = TrendAnalyzer()
        summary = analyzer.analyze(buf, {"T01": SensorType.TEMP})
        assert abs(summary.channels[0].slope) < 0.001
        assert summary.channels[0].std < 0.001


class TestFFTPredictor:
    def test_predict_sinusoidal(self):
        values = [math.sin(i * 0.1) for i in range(200)]
        pred = FFTPredictor(top_k=4)
        result = pred.predict(values, horizon=20)
        assert len(result) == 20
        assert all(isinstance(v, float) for v in result)

    def test_predict_short_series(self):
        pred = FFTPredictor()
        result = pred.predict([1.0, 2.0], horizon=5)
        assert len(result) == 5

    def test_predict_constant(self):
        pred = FFTPredictor()
        result = pred.predict([5.0] * 100, horizon=10)
        for v in result:
            assert abs(v - 5.0) < 0.5

    def test_predict_empty(self):
        pred = FFTPredictor()
        result = pred.predict([], horizon=5)
        assert result == [0.0] * 5


class TestAnomalyAggregator:
    def _make_channel(self, sid, stype, score, is_anomaly):
        return ProcessedChannel(
            sensor_id=sid, sensor_type=stype,
            raw_value=0, kalman_value=0, ewma_value=0,
            anomaly_score=score, is_anomaly=is_anomaly, ts=BASE_TS,
        )

    def test_composite_score(self):
        agg = AnomalyAggregator()
        channels = [
            self._make_channel("T01", SensorType.TEMP, 0.5, True),
            self._make_channel("A01", SensorType.ACTIVITY, 0.8, True),
        ]
        result = agg.aggregate(channels)
        assert 0.0 < result.composite_score <= 1.0
        assert len(result.top_anomalies) == 2
        assert result.top_anomalies[0].score == 0.8

    def test_empty_channels(self):
        agg = AnomalyAggregator()
        result = agg.aggregate([])
        assert result.composite_score == 0.0


class TestHaversine:
    def test_same_point(self):
        assert haversine(37.5665, 126.978, 37.5665, 126.978) == 0.0

    def test_known_distance(self):
        dist = haversine(37.5665, 126.978, 37.5700, 126.982)
        assert 400 < dist < 600

    def test_seoul_to_busan(self):
        dist = haversine(37.5665, 126.978, 35.1796, 129.0756)
        assert 300_000 < dist < 350_000


class TestGPSAnalyzer:
    def test_at_home(self):
        home = HomeLocation(lat=37.5665, lon=126.978)
        analyzer = GPSAnalyzer(home)
        analyzer.update(MobileGPS(
            subject_id="U001", lat=37.5665, lon=126.978,
            ts=BASE_TS, device_id="mob_001",
        ))
        result = analyzer.evaluate(BASE_TS + 100)
        assert result is not None
        assert result.is_no_return_risk is False
        assert result.distance_from_home_m < 10

    def test_no_return_risk(self):
        home = HomeLocation(lat=37.5665, lon=126.978)
        analyzer = GPSAnalyzer(home, distance_threshold_m=500, duration_threshold_sec=3600)
        analyzer.update(MobileGPS(
            subject_id="U001", lat=37.5665, lon=126.978,
            ts=BASE_TS, device_id="mob_001",
        ))
        analyzer.update(MobileGPS(
            subject_id="U001", lat=37.60, lon=127.02,
            ts=BASE_TS + 4000, device_id="mob_001",
        ))
        result = analyzer.evaluate(BASE_TS + 8000)
        assert result is not None
        assert result.is_no_return_risk is True
        assert result.distance_from_home_m > 500

    def test_no_gps_returns_none(self):
        home = HomeLocation(lat=37.5665, lon=126.978)
        analyzer = GPSAnalyzer(home)
        assert analyzer.evaluate(BASE_TS) is None

    def test_first_gps_already_far(self):
        home = HomeLocation(lat=37.5665, lon=126.978)
        analyzer = GPSAnalyzer(home, distance_threshold_m=500, duration_threshold_sec=3600)
        analyzer.update(MobileGPS(
            subject_id="U001", lat=37.60, lon=127.02,
            ts=BASE_TS, device_id="mob_001",
        ))
        result = analyzer.evaluate(BASE_TS + 4000)
        assert result is not None
        assert result.is_no_return_risk is True


class TestGradePredictor:
    def test_watch_on_normal(self):
        from shared.schemas import TrendSummary, AnomalySummary
        gp = GradePredictor()
        pred = gp.predict(
            SituationGrade.WATCH,
            TrendSummary(window_days=7, channels=[]),
            AnomalySummary(composite_score=0.0, top_anomalies=[]),
            [],
        )
        assert pred.predicted_grade == SituationGrade.WATCH
        assert pred.predicted_risk_score < 0.3

    def test_danger_on_high_anomaly(self):
        from shared.schemas import TrendSummary, AnomalySummary, ChannelTrend
        gp = GradePredictor()
        pred = gp.predict(
            SituationGrade.DANGER,
            TrendSummary(window_days=7, channels=[
                ChannelTrend(sensor_type=SensorType.TEMP, mean=24, std=2, slope=-0.05, min=20, max=28),
            ]),
            AnomalySummary(composite_score=0.9, top_anomalies=[]),
            [CrossScenarioResult(
                scenario_type="heatstroke", triggered=True,
                confidence=0.9, contributing_sensors=["T01"],
            )],
            predicted_deviation=0.15,
        )
        assert pred.predicted_grade == SituationGrade.DANGER
        assert pred.predicted_risk_score >= 0.6

    def test_recovery_from_danger(self):
        from shared.schemas import TrendSummary, AnomalySummary
        gp = GradePredictor()
        pred = gp.predict(
            SituationGrade.DANGER,
            TrendSummary(window_days=7, channels=[]),
            AnomalySummary(composite_score=0.0, top_anomalies=[]),
            [],
        )
        assert pred.predicted_grade == SituationGrade.WATCH
        assert pred.predicted_risk_score < 0.3


class TestEdgePredictionPipeline:
    def _make_preprocessed(self, channels, grade, ts):
        return EdgePreprocessedData(
            subject_id="U001", gateway_id="gw_001",
            channels=channels, situation_grade=grade,
            grade_reason="test", cycle_ts=ts,
        )

    def _make_channel(self, sid, stype, value, ts):
        return ProcessedChannel(
            sensor_id=sid, sensor_type=stype,
            raw_value=value, kalman_value=value, ewma_value=value,
            anomaly_score=0.0, is_anomaly=False, ts=ts,
        )

    def test_record_and_build(self):
        pipeline = EdgePredictionPipeline(
            home_location=HomeLocation(lat=37.5665, lon=126.978),
        )
        for i in range(30):
            ts = BASE_TS + i * 600
            channels = [
                self._make_channel("T01", SensorType.TEMP, 24.0 + i * 0.1, ts),
                self._make_channel("A01", SensorType.ACTIVITY, 1.0, ts),
            ]
            preprocessed = self._make_preprocessed(channels, SituationGrade.WATCH, ts)
            pipeline.record(preprocessed)

        last = self._make_preprocessed(
            [self._make_channel("T01", SensorType.TEMP, 27.0, BASE_TS + 18000)],
            SituationGrade.WATCH,
            BASE_TS + 18000,
        )
        payload = pipeline.build_payload(last, TransmissionTrigger.PERIODIC)

        assert isinstance(payload, EdgeCompactPayload)
        assert payload.subject_id == "U001"
        assert payload.transmission_trigger == TransmissionTrigger.PERIODIC
        assert len(payload.trend_summary.channels) > 0
        assert payload.prediction.predicted_grade == SituationGrade.WATCH

    def test_gps_integration(self):
        pipeline = EdgePredictionPipeline(
            home_location=HomeLocation(lat=37.5665, lon=126.978),
        )
        pipeline.update_gps(MobileGPS(
            subject_id="U001", lat=37.60, lon=127.02,
            ts=BASE_TS, device_id="mob_001",
        ))
        channels = [self._make_channel("T01", SensorType.TEMP, 24.0, BASE_TS)]
        preprocessed = self._make_preprocessed(channels, SituationGrade.WATCH, BASE_TS + 8000)
        pipeline.record(preprocessed)
        payload = pipeline.build_payload(preprocessed, TransmissionTrigger.PERIODIC)
        assert payload.gps_summary is not None
        assert payload.gps_summary.distance_from_home_m > 100

    def test_expert_events_included(self):
        from shared.schemas import ExpertEvent
        pipeline = EdgePredictionPipeline()
        channels = [self._make_channel("T01", SensorType.TEMP, 24.0, BASE_TS)]
        preprocessed = self._make_preprocessed(channels, SituationGrade.DANGER, BASE_TS)
        pipeline.record(preprocessed)

        event = ExpertEvent(
            event_type="fire", state="smoke",
            triggered_ts=BASE_TS, action_taken="dispatch_119",
        )
        payload = pipeline.build_payload(
            preprocessed, TransmissionTrigger.EXPERT_EVENT, [event],
        )
        assert len(payload.expert_events) == 1
        assert payload.expert_events[0].event_type == "fire"

    def test_payload_schema_valid(self):
        pipeline = EdgePredictionPipeline()
        channels = [self._make_channel("T01", SensorType.TEMP, 24.0, BASE_TS)]
        preprocessed = self._make_preprocessed(channels, SituationGrade.WATCH, BASE_TS)
        pipeline.record(preprocessed)
        payload = pipeline.build_payload(preprocessed, TransmissionTrigger.PERIODIC)
        dumped = payload.model_dump()
        restored = EdgeCompactPayload(**dumped)
        assert restored.subject_id == payload.subject_id
