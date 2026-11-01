import pytest
from pydantic import ValidationError

from shared.schemas import (
    ActivityReading,
    ChannelTrend,
    CoreInferenceResult,
    DoorReading,
    EdgeCompactPayload,
    EdgePreprocessedData,
    EmergencyCallReading,
    ExpertEvent,
    FireAlertReading,
    HumidityReading,
    IlluminanceReading,
    MobileGPS,
    MobileRiskAssessment,
    ModelOutputs,
    MotionReading,
    ProcessedChannel,
    SensorType,
    SituationGrade,
    TemperatureReading,
    TransmissionTrigger,
    TrendSummary,
    AnomalySummary,
    AnomalyRank,
    EdgePrediction,
)
from edge_ai.tests.conftest import (
    BASE_TS,
    make_temp,
    make_humidity,
    make_lux,
    make_activity,
    make_pir,
    make_fire,
    make_emergency,
    make_door,
    make_gps,
)


class TestSensorReadings:
    def test_temperature(self):
        r = make_temp(BASE_TS, 28.3)
        assert r.sensor_type == SensorType.TEMP
        assert r.value == 28.3
        assert r.unit == "C"

    def test_humidity(self):
        r = make_humidity(BASE_TS, 65.2)
        assert r.sensor_type == SensorType.HUMI
        assert r.unit == "%"

    def test_illuminance(self):
        r = make_lux(BASE_TS, 210.0)
        assert r.sensor_type == SensorType.LUX
        assert r.unit == "lx"

    def test_activity(self):
        r = make_activity(BASE_TS, value=1, count=18)
        assert r.sensor_type == SensorType.ACTIVITY
        assert r.period_end == BASE_TS + 60

    def test_pir(self):
        r = make_pir(BASE_TS, value=1, count=22)
        assert r.sensor_type == SensorType.PIR

    def test_fire_alert(self):
        r = make_fire(BASE_TS, "smoke")
        assert r.sensor_type == SensorType.FIRE
        assert r.state == "smoke"

    def test_emergency_call(self):
        r = make_emergency(BASE_TS, "red")
        assert r.sensor_type == SensorType.EMERGENCY

    def test_door(self):
        r = make_door(BASE_TS, "open", "out")
        assert r.sensor_type == SensorType.DOOR
        assert r.direction == "out"

    def test_mobile_gps(self):
        g = make_gps(BASE_TS, 37.5665, 126.9780)
        assert g.subject_id == "U001"

    def test_mobile_risk(self):
        r = MobileRiskAssessment(
            subject_id="U001", risk_level="warning",
            fall=True, no_return=False, usage_anomaly=False,
            long_inactivity=False, ts=BASE_TS, device_id="mob_001",
        )
        assert r.fall is True

    def test_serialization_roundtrip(self):
        original = make_temp(BASE_TS, 28.3)
        dumped = original.model_dump()
        restored = TemperatureReading(**dumped)
        assert restored == original


class TestEdgePreprocessedData:
    def test_minimal(self):
        data = EdgePreprocessedData(
            subject_id="U001", gateway_id="gw_001",
            channels=[], situation_grade=SituationGrade.WATCH,
            grade_reason="normal", cycle_ts=BASE_TS,
        )
        assert data.situation_grade == SituationGrade.WATCH
        assert data.cross_scenarios == []

    def test_with_channels(self):
        ch = ProcessedChannel(
            sensor_id="T01", sensor_type=SensorType.TEMP,
            raw_value=28.3, kalman_value=28.2, ewma_value=28.1,
            anomaly_score=0.1, is_anomaly=False, ts=BASE_TS,
        )
        data = EdgePreprocessedData(
            subject_id="U001", gateway_id="gw_001",
            channels=[ch], situation_grade=SituationGrade.WATCH,
            grade_reason="normal", cycle_ts=BASE_TS,
        )
        assert len(data.channels) == 1


class TestEdgeCompactPayload:
    def test_full_payload(self):
        payload = EdgeCompactPayload(
            subject_id="U001", gateway_id="gw_001",
            window_start=BASE_TS - 3600, window_end=BASE_TS,
            transmission_trigger=TransmissionTrigger.PERIODIC,
            edge_grade=SituationGrade.WATCH,
            trend_summary=TrendSummary(
                window_days=7,
                channels=[ChannelTrend(
                    sensor_type=SensorType.TEMP,
                    mean=24.0, std=2.0, slope=0.01, min=20.0, max=28.0,
                )],
            ),
            anomaly_summary=AnomalySummary(
                composite_score=0.1,
                top_anomalies=[AnomalyRank(
                    sensor_type=SensorType.TEMP, score=0.1, sensor_id="T01",
                )],
            ),
            prediction=EdgePrediction(
                predicted_grade=SituationGrade.WATCH,
                predicted_risk_score=0.05, horizon_minutes=60,
            ),
            sent_ts=BASE_TS,
        )
        assert payload.transmission_trigger == TransmissionTrigger.PERIODIC

    def test_anomaly_score_bounds(self):
        with pytest.raises(ValidationError):
            AnomalySummary(composite_score=1.5, top_anomalies=[])
