from __future__ import annotations

import time
from typing import Iterator

import pytest

from shared.schemas import (
    ActivityReading,
    CognitiveGroup,
    ContextSeverity,
    DoorReading,
    EmergencyCallReading,
    ExternalContext,
    FireAlertReading,
    GatewayConfig,
    HomeLocation,
    HumidityReading,
    IlluminanceReading,
    MobileGPS,
    MobileRiskAssessment,
    MotionReading,
    PersonalProfile,
    SensorType,
    TemperatureReading,
)

BASE_TS = 1742961600.0
GW_ID = "gw_001"
SUBJECT_ID = "U001"


@pytest.fixture
def base_ts() -> float:
    return BASE_TS


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig(
        gateway_id=GW_ID,
        subject_id=SUBJECT_ID,
        device_mappings={
            "T01": SensorType.TEMP,
            "H01": SensorType.HUMI,
            "L01": SensorType.LUX,
            "A01": SensorType.ACTIVITY,
            "P01": SensorType.PIR,
            "F01": SensorType.FIRE,
            "E01": SensorType.EMERGENCY,
            "D01": SensorType.DOOR,
        },
        home_location=HomeLocation(lat=37.5665, lon=126.9780),
    )


@pytest.fixture
def normal_profile() -> PersonalProfile:
    return PersonalProfile(
        subject_id=SUBJECT_ID,
        age=78,
        cognitive_group=CognitiveGroup.CN,
        mobility_impaired=False,
        home_location=HomeLocation(lat=37.5665, lon=126.9780),
        enrollment_date=BASE_TS - 86400 * 30,
    )


@pytest.fixture
def impaired_profile() -> PersonalProfile:
    return PersonalProfile(
        subject_id=SUBJECT_ID,
        age=85,
        cognitive_group=CognitiveGroup.MCI,
        mobility_impaired=True,
        sleep_disorder=True,
        home_location=HomeLocation(lat=37.5665, lon=126.9780),
        enrollment_date=BASE_TS - 86400 * 30,
    )


def make_temp(ts: float, value: float, seq: int = 1) -> TemperatureReading:
    return TemperatureReading(
        sensor_id="T01", device_id=GW_ID, seq=seq, ts=ts, value=value,
    )


def make_humidity(ts: float, value: float, seq: int = 1) -> HumidityReading:
    return HumidityReading(
        sensor_id="H01", device_id=GW_ID, seq=seq, ts=ts, value=value,
    )


def make_lux(ts: float, value: float, seq: int = 1) -> IlluminanceReading:
    return IlluminanceReading(
        sensor_id="L01", device_id=GW_ID, seq=seq, ts=ts, value=value,
    )


def make_activity(
    ts: float, value: int = 1, count: int = 18, seq: int = 1,
) -> ActivityReading:
    return ActivityReading(
        sensor_id="A01", device_id=GW_ID, seq=seq, ts=ts,
        period_start=ts, period_end=ts + 60,
        value=value, count=count, max=1, min=0, status=0,
    )


def make_pir(
    ts: float, value: int = 1, count: int = 22, seq: int = 1,
) -> MotionReading:
    return MotionReading(
        sensor_id="P01", device_id="pir_living", seq=seq, ts=ts,
        period_start=ts, period_end=ts + 60,
        value=value, count=count, max=1, min=0, status=0,
    )


def make_fire(ts: float, state: str = "smoke", seq: int = 1) -> FireAlertReading:
    return FireAlertReading(
        sensor_id="F01", device_id="fire_kitchen", seq=seq, ts=ts,
        state=state, status=0,
    )


def make_emergency(ts: float, state: str = "red", seq: int = 1) -> EmergencyCallReading:
    return EmergencyCallReading(
        sensor_id="E01", device_id="emg_bathroom", seq=seq, ts=ts,
        state=state, status=0,
    )


def make_door(
    ts: float, state: str = "open", direction: str = "out", seq: int = 1,
) -> DoorReading:
    return DoorReading(
        sensor_id="D01", device_id="door_front", seq=seq, ts=ts,
        state=state, direction=direction, status=0,
    )


def make_gps(ts: float, lat: float, lon: float) -> MobileGPS:
    return MobileGPS(
        subject_id=SUBJECT_ID, lat=lat, lon=lon, ts=ts, device_id="mob_001",
    )


def make_heat_alert(ts: float, severity: ContextSeverity = ContextSeverity.WARNING) -> ExternalContext:
    return ExternalContext(
        context_id="heat_001",
        context_type="weather_alert",
        severity=severity,
        region_code="1100000000",
        message="heat wave warning",
        parameters={"type": "heat"},
        effective_from=ts,
        effective_until=ts + 86400,
        received_ts=ts,
    )


def generate_normal_day(start_ts: float, duration_hours: int = 24, start_hour: int = 8) -> list:
    readings = []
    ts = start_ts
    end = start_ts + duration_hours * 3600
    seq = 1

    while ts < end:
        hour = (start_hour + int((ts - start_ts) / 3600)) % 24

        if 7 <= hour <= 22:
            temp = 24.0 + (hour - 12) * 0.3
            activity_val = 1
            pir_val = 1
            lux = 200.0 + hour * 10
        else:
            temp = 22.0
            activity_val = 0
            pir_val = 0
            lux = 5.0

        readings.append(make_temp(ts, temp, seq))
        readings.append(make_humidity(ts, 55.0 + hour * 0.5, seq))
        readings.append(make_lux(ts, lux, seq))

        if seq % 10 == 0 or activity_val > 0:
            readings.append(make_activity(ts, activity_val, seq=seq))
            readings.append(make_pir(ts, pir_val, seq=seq))

        ts += 600
        seq += 1

    return readings


def generate_heatstroke_scenario(start_ts: float) -> list:
    readings = []
    seq = 1

    for i in range(30):
        ts = start_ts + i * 60
        readings.append(make_temp(ts, 24.0 + i * 0.1, seq))
        readings.append(make_activity(ts, 1, seq=seq))
        readings.append(make_pir(ts, 1, seq=seq))
        seq += 1

    for i in range(40):
        ts = start_ts + 1800 + i * 60
        readings.append(make_temp(ts, 36.0 + i * 0.05, seq))
        readings.append(make_activity(ts, 0, count=0, seq=seq))
        readings.append(make_pir(ts, 0, count=0, seq=seq))
        seq += 1

    return readings


def generate_fall_scenario(start_ts: float) -> list:
    readings = []
    seq = 1

    for i in range(20):
        ts = start_ts + i * 60
        readings.append(make_temp(ts, 24.0, seq))
        readings.append(make_activity(ts, 1, seq=seq))
        readings.append(make_pir(ts, 1, seq=seq))
        seq += 1

    for i in range(35):
        ts = start_ts + 1200 + i * 60
        readings.append(make_temp(ts, 24.0, seq))
        readings.append(make_activity(ts, 0, count=0, seq=seq))
        readings.append(make_pir(ts, 0, count=0, seq=seq))
        seq += 1

    return readings


def generate_hypothermia_scenario(start_ts: float) -> list:
    readings = []
    seq = 1

    for i in range(30):
        ts = start_ts + i * 60
        readings.append(make_temp(ts, 20.0 - i * 0.2, seq))
        readings.append(make_activity(ts, 1, seq=seq))
        readings.append(make_pir(ts, 1, seq=seq))
        seq += 1

    for i in range(30):
        ts = start_ts + 1800 + i * 60
        readings.append(make_temp(ts, 13.0 - i * 0.05, seq))
        readings.append(make_activity(ts, 0, count=0, seq=seq))
        readings.append(make_pir(ts, 0, count=0, seq=seq))
        seq += 1

    return readings


def generate_fire_event(ts: float) -> list:
    return [
        make_fire(ts, "smoke"),
    ]


def generate_fire_cancelled(ts: float, cancel_delay: float = 10.0) -> list:
    return [
        make_fire(ts, "smoke"),
        make_fire(ts + cancel_delay, "normal"),
    ]


def generate_emergency_event(ts: float) -> list:
    return [
        make_emergency(ts, "red"),
    ]
