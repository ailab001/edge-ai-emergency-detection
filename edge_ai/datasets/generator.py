"""
Test dataset generator for Edge AI local testing.

Generates realistic sensor data streams matching the proposal's JSON format.
Covers: normal daily patterns, heatstroke, fall, hypothermia, fire, emergency,
night wandering, and no-return scenarios.

Usage:
    python -m edge_ai.datasets.generator
    python -m edge_ai.datasets.generator --output ./my_datasets
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

BASE_TS = 1742961600.0
MINUTE = 60
HOUR = 3600
DAY = 86400


def _noise(base: float, scale: float = 0.5) -> float:
    return base + random.gauss(0, scale)


def _make_temp(ts, value, seq, device_id="gw_001"):
    return {
        "sensor_id": "T01", "sensor_type": "temp",
        "value": round(value, 1), "unit": "C",
        "ts": ts, "device_id": device_id, "seq": seq,
    }


def _make_humi(ts, value, seq, device_id="gw_001"):
    return {
        "sensor_id": "H01", "sensor_type": "humi",
        "value": round(value, 1), "unit": "%",
        "ts": ts, "device_id": device_id, "seq": seq,
    }


def _make_lux(ts, value, seq, device_id="gw_001"):
    return {
        "sensor_id": "L01", "sensor_type": "lux",
        "value": round(max(0, value), 1), "unit": "lx",
        "ts": ts, "device_id": device_id, "seq": seq,
    }


def _make_activity(ts, value, count, seq, device_id="gw_001"):
    return {
        "sensor_id": "A01", "sensor_type": "activity",
        "period_start": ts, "period_end": ts + 60,
        "value": value, "count": count,
        "max": 1 if value > 0 else 0, "min": 0,
        "device_id": device_id, "seq": seq, "status": 0, "ts": ts,
    }


def _make_pir(ts, value, count, seq, device_id="pir_living"):
    return {
        "sensor_id": "P01", "sensor_type": "pir",
        "period_start": ts, "period_end": ts + 60,
        "value": value, "count": count,
        "max": 1 if value > 0 else 0, "min": 0,
        "device_id": device_id, "seq": seq, "status": 0, "ts": ts,
    }


def _make_fire(ts, state, seq, device_id="fire_kitchen"):
    return {
        "sensor_id": "F01", "sensor_type": "fire",
        "state": state, "ts": ts,
        "device_id": device_id, "seq": seq, "status": 0,
    }


def _make_emergency(ts, state, seq, device_id="emg_bathroom"):
    return {
        "sensor_id": "E01", "sensor_type": "emergency",
        "state": state, "ts": ts,
        "device_id": device_id, "seq": seq, "status": 0,
    }


def _make_door(ts, state, direction, seq, device_id="door_front"):
    return {
        "sensor_id": "D01", "sensor_type": "door",
        "state": state, "direction": direction, "ts": ts,
        "device_id": device_id, "seq": seq, "status": 0,
    }


def _make_mobile_risk(ts, risk_level, fall=False, no_return=False):
    return {
        "subject_id": "U001", "risk_level": risk_level,
        "fall": fall, "no_return": no_return,
        "usage_anomaly": False, "long_inactivity": False,
        "ts": ts, "device_id": "mob_001",
    }


def _make_gps(ts, lat, lon):
    return {
        "subject_id": "U001", "lat": lat, "lon": lon,
        "ts": ts, "device_id": "mob_001",
    }


def _hour_of(ts, start_ts):
    return int((ts - start_ts) / HOUR) % 24


def generate_normal_7days(start_ts: float) -> dict:
    readings = []
    mobile = []
    seq = 1

    for day in range(7):
        day_start = start_ts + day * DAY

        for minute_offset in range(0, 24 * 60, 1):
            ts = day_start + minute_offset * MINUTE
            hour = (minute_offset // 60) % 24

            if minute_offset % 10 == 0:
                if 7 <= hour <= 22:
                    temp = _noise(23.0 + math.sin((hour - 6) * math.pi / 16) * 4, 0.3)
                    humi = _noise(55.0 - math.sin((hour - 6) * math.pi / 16) * 10, 1.0)
                    lux = _noise(150 + math.sin((hour - 6) * math.pi / 16) * 200, 10)
                else:
                    temp = _noise(21.0, 0.2)
                    humi = _noise(60.0, 1.0)
                    lux = _noise(3.0, 1.0)
                readings.append(_make_temp(ts, temp, seq))
                readings.append(_make_humi(ts, humi, seq))
                readings.append(_make_lux(ts, lux, seq))

            if 7 <= hour <= 22:
                active = random.random() < 0.7
                readings.append(_make_activity(ts, 1 if active else 0, random.randint(5, 30) if active else 0, seq))
                readings.append(_make_pir(ts, 1 if active else 0, random.randint(3, 25) if active else 0, seq))
            elif minute_offset % 5 == 0:
                readings.append(_make_activity(ts, 0, 0, seq))
                readings.append(_make_pir(ts, 0, 0, seq))

            if hour == 8 and minute_offset % 60 == 0:
                readings.append(_make_door(ts, "open", "out", seq))
                readings.append(_make_door(ts + 5, "normal", "-", seq + 1))

            if hour == 17 and minute_offset % 60 == 0:
                readings.append(_make_door(ts, "open", "in", seq))
                readings.append(_make_door(ts + 5, "normal", "-", seq + 1))

            seq += 1

        for h in [9, 12, 15, 18]:
            gps_ts = day_start + h * HOUR
            mobile.append(_make_gps(gps_ts, _noise(37.5665, 0.001), _noise(126.9780, 0.001)))

    return {
        "scenario": "normal_7days",
        "description": "7-day normal daily pattern. Activity during 07-22h, sleep at night.",
        "expected_grade": "watch",
        "readings": readings,
        "mobile": mobile,
    }


def generate_heatstroke(start_ts: float) -> dict:
    readings = []
    seq = 1

    for i in range(60):
        ts = start_ts + i * MINUTE
        readings.append(_make_temp(ts, _noise(24.0 + i * 0.05, 0.2), seq))
        readings.append(_make_humi(ts, _noise(55.0, 1.0), seq))
        readings.append(_make_activity(ts, 1, random.randint(10, 25), seq))
        readings.append(_make_pir(ts, 1, random.randint(8, 20), seq))
        seq += 1

    for i in range(50):
        ts = start_ts + 3600 + i * MINUTE
        temp = 35.5 + i * 0.08
        readings.append(_make_temp(ts, _noise(temp, 0.3), seq))
        readings.append(_make_humi(ts, _noise(70.0, 2.0), seq))
        readings.append(_make_activity(ts, 0, 0, seq))
        readings.append(_make_pir(ts, 0, 0, seq))
        seq += 1

    return {
        "scenario": "heatstroke",
        "description": "High temp (>35C) + inactivity >= 30min. Simulates summer heatstroke risk.",
        "expected_grade": "caution or danger",
        "trigger_conditions": {
            "temp_threshold": 35.0,
            "inactivity_minutes": 30,
        },
        "readings": readings,
        "mobile": [],
    }


def generate_fall(start_ts: float) -> dict:
    readings = []
    seq = 1

    for i in range(20):
        ts = start_ts + i * MINUTE
        readings.append(_make_temp(ts, _noise(24.0, 0.2), seq))
        readings.append(_make_activity(ts, 1, random.randint(10, 20), seq))
        readings.append(_make_pir(ts, 1, random.randint(8, 15), seq))
        seq += 1

    readings.append(_make_door(start_ts + 20 * MINUTE, "open", "in", seq))
    seq += 1

    for i in range(45):
        ts = start_ts + 20 * MINUTE + i * MINUTE
        readings.append(_make_temp(ts, _noise(24.0, 0.2), seq))
        readings.append(_make_activity(ts, 0, 0, seq))
        readings.append(_make_pir(ts, 0, 0, seq))
        seq += 1

    return {
        "scenario": "fall",
        "description": "Bathroom entry + PIR/activity silent >= 30min. Simulates fall in bathroom.",
        "expected_grade": "caution or danger",
        "trigger_conditions": {
            "inactivity_minutes": 30,
            "pir_silent": True,
        },
        "readings": readings,
        "mobile": [],
    }


def generate_hypothermia(start_ts: float) -> dict:
    readings = []
    seq = 1

    for i in range(60):
        ts = start_ts + i * MINUTE
        temp = 20.0 - i * 0.12
        readings.append(_make_temp(ts, _noise(temp, 0.3), seq))
        readings.append(_make_humi(ts, _noise(65.0, 1.0), seq))

        if i < 30:
            readings.append(_make_activity(ts, 1, random.randint(8, 20), seq))
            readings.append(_make_pir(ts, 1, random.randint(5, 15), seq))
        else:
            activity_val = 1 if random.random() < max(0, 0.3 - (i - 30) * 0.01) else 0
            readings.append(_make_activity(ts, activity_val, random.randint(0, 3) if activity_val else 0, seq))
            readings.append(_make_pir(ts, activity_val, random.randint(0, 2) if activity_val else 0, seq))
        seq += 1

    for i in range(30):
        ts = start_ts + 3600 + i * MINUTE
        readings.append(_make_temp(ts, _noise(12.5 - i * 0.05, 0.2), seq))
        readings.append(_make_humi(ts, _noise(70.0, 1.5), seq))
        readings.append(_make_activity(ts, 0, 0, seq))
        readings.append(_make_pir(ts, 0, 0, seq))
        seq += 1

    return {
        "scenario": "hypothermia",
        "description": "Indoor temp <= 15C + activity drop. Simulates winter hypothermia risk.",
        "expected_grade": "caution or danger",
        "trigger_conditions": {
            "temp_threshold_low": 15.0,
            "activity_drop_ratio": 0.3,
        },
        "readings": readings,
        "mobile": [],
    }


def generate_fire(start_ts: float) -> dict:
    normal_readings = []
    seq = 1
    for i in range(10):
        ts = start_ts + i * MINUTE
        normal_readings.append(_make_temp(ts, _noise(24.0, 0.2), seq))
        normal_readings.append(_make_activity(ts, 1, 15, seq))
        seq += 1

    fire_ts = start_ts + 10 * MINUTE
    fire_readings = [_make_fire(fire_ts, "smoke", seq)]

    return {
        "scenario": "fire_no_cancel",
        "description": "Smoke detected, no cancel within 15s. Expects 119 dispatch.",
        "expected_grade": "danger",
        "expected_action": "dispatch_119",
        "cancel_window_sec": 15.0,
        "readings": normal_readings + fire_readings,
        "mobile": [],
    }


def generate_fire_cancelled(start_ts: float) -> dict:
    fire_ts = start_ts
    readings = [
        _make_fire(fire_ts, "smoke", 1),
        _make_fire(fire_ts + 10.0, "normal", 2),
    ]
    return {
        "scenario": "fire_cancelled",
        "description": "Smoke detected, cancelled within 10s. No dispatch.",
        "expected_grade": "watch",
        "expected_action": "monitor",
        "readings": readings,
        "mobile": [],
    }


def generate_emergency(start_ts: float) -> dict:
    readings = [_make_emergency(start_ts, "red", 1)]
    return {
        "scenario": "emergency_call",
        "description": "Emergency button pressed. No cancel within 15s. Expects 119 dispatch.",
        "expected_grade": "danger",
        "expected_action": "dispatch_119",
        "readings": readings,
        "mobile": [],
    }


def generate_night_wandering(start_ts: float) -> dict:
    readings = []
    seq = 1

    night_start = start_ts + 23 * HOUR

    for i in range(60):
        ts = night_start + i * MINUTE
        readings.append(_make_temp(ts, _noise(22.0, 0.2), seq))
        readings.append(_make_lux(ts, _noise(2.0, 0.5), seq))

        if i < 10:
            readings.append(_make_activity(ts, 0, 0, seq))
            readings.append(_make_pir(ts, 0, 0, seq))
        else:
            readings.append(_make_activity(ts, 1, random.randint(3, 10), seq))
            readings.append(_make_pir(ts, 1, random.randint(2, 8), seq))

        if i == 15:
            readings.append(_make_door(ts, "open", "out", seq))
        if i == 45:
            readings.append(_make_door(ts, "open", "in", seq))
        seq += 1

    return {
        "scenario": "night_wandering",
        "description": "Night-time (23-01h) PIR activation + door open/close. Possible wandering. (WanderingChecker not yet implemented)",
        "expected_grade": "watch",
        "readings": readings,
        "mobile": [],
    }


def generate_no_return(start_ts: float) -> dict:
    readings = []
    mobile = []
    seq = 1

    for i in range(60):
        ts = start_ts + 8 * HOUR + i * MINUTE
        readings.append(_make_temp(ts, _noise(24.0, 0.3), seq))
        readings.append(_make_activity(ts, 1, random.randint(10, 25), seq))
        seq += 1

    out_ts = start_ts + 9 * HOUR
    readings.append(_make_door(out_ts, "open", "out", seq))
    seq += 1

    for i in range(6 * 60):
        ts = out_ts + i * MINUTE
        readings.append(_make_temp(ts, _noise(24.0, 0.2), seq))
        readings.append(_make_activity(ts, 0, 0, seq))
        readings.append(_make_pir(ts, 0, 0, seq))
        seq += 1

    home_lat, home_lon = 37.5665, 126.9780
    for i in range(36):
        gps_ts = out_ts + i * 10 * MINUTE
        dist_km = 0.5 + i * 0.3
        angle = math.radians(45 + random.gauss(0, 5))
        lat = home_lat + (dist_km / 111.32) * math.cos(angle)
        lon = home_lon + (dist_km / (111.32 * math.cos(math.radians(home_lat)))) * math.sin(angle)
        mobile.append(_make_gps(gps_ts, round(lat, 6), round(lon, 6)))

    mobile.append(_make_mobile_risk(out_ts + 4 * HOUR, "warning", no_return=True))

    return {
        "scenario": "no_return",
        "description": "Left home at 09:00, GPS shows increasing distance, no return by 15:00.",
        "expected_grade": "caution or danger",
        "trigger_conditions": {
            "home_lat": home_lat,
            "home_lon": home_lon,
            "max_distance_km": 10.0,
        },
        "readings": readings,
        "mobile": mobile,
    }


def generate_profiles() -> list[dict]:
    return [
        {
            "subject_id": "U001",
            "age": 78,
            "cognitive_group": "CN",
            "mobility_impaired": False,
            "sleep_disorder": False,
            "medications": [],
            "home_location": {"lat": 37.5665, "lon": 126.9780, "address": None},
            "living_alone": True,
            "emergency_contacts": [
                {"name": "Kim", "phone": "010-1234-5678", "relationship": "son"}
            ],
            "custom_thresholds": None,
            "enrollment_date": BASE_TS - 30 * DAY,
            "notes": None,
        },
        {
            "subject_id": "U002",
            "age": 85,
            "cognitive_group": "MCI",
            "mobility_impaired": True,
            "sleep_disorder": True,
            "medications": ["donepezil", "amlodipine"],
            "home_location": {"lat": 37.5700, "lon": 126.9820, "address": None},
            "living_alone": True,
            "emergency_contacts": [
                {"name": "Park", "phone": "010-9876-5432", "relationship": "daughter"}
            ],
            "custom_thresholds": {"heatstroke.temp_high": 33.0},
            "enrollment_date": BASE_TS - 60 * DAY,
            "notes": "mobility_impaired -> hypothermia threshold raised to 18C",
        },
        {
            "subject_id": "U003",
            "age": 82,
            "cognitive_group": "Dem",
            "mobility_impaired": False,
            "sleep_disorder": True,
            "medications": ["memantine"],
            "home_location": {"lat": 37.5600, "lon": 126.9750, "address": None},
            "living_alone": True,
            "emergency_contacts": [
                {"name": "Lee", "phone": "010-5555-6666", "relationship": "son"}
            ],
            "custom_thresholds": {"fall.inactivity_minutes": 20.0},
            "enrollment_date": BASE_TS - 90 * DAY,
            "notes": "dementia group -> stricter fall detection",
        },
    ]


def generate_contexts() -> list[dict]:
    return [
        {
            "context_id": "heat_001",
            "context_type": "weather_alert",
            "severity": "warning",
            "region_code": "1100000000",
            "message": "heat wave warning issued for Seoul metropolitan area",
            "parameters": {"type": "heat", "max_temp_forecast": 38},
            "effective_from": BASE_TS,
            "effective_until": BASE_TS + 3 * DAY,
            "threshold_adjustments": [],
            "received_ts": BASE_TS,
        },
        {
            "context_id": "cold_001",
            "context_type": "weather_alert",
            "severity": "watch",
            "region_code": "1100000000",
            "message": "cold wave watch issued",
            "parameters": {"type": "cold", "min_temp_forecast": -15},
            "effective_from": BASE_TS,
            "effective_until": BASE_TS + 2 * DAY,
            "threshold_adjustments": [
                {
                    "target_sensor_type": "temp",
                    "target_scenario": "hypothermia",
                    "parameter_name": "temp_low",
                    "original_value": 15.0,
                    "adjusted_value": 17.0,
                    "reason": "cold_wave_watch",
                }
            ],
            "received_ts": BASE_TS,
        },
    ]


def generate_gateway_config() -> dict:
    return {
        "gateway_id": "gw_001",
        "subject_id": "U001",
        "device_mappings": {
            "T01": "temp",
            "H01": "humi",
            "L01": "lux",
            "A01": "activity",
            "P01": "pir",
            "F01": "fire",
            "E01": "emergency",
            "D01": "door",
        },
        "home_location": {"lat": 37.5665, "lon": 126.9780},
    }


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {path} ({len(json.dumps(data)):,} bytes)")


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    random.seed(42)

    print("Generating test datasets...\n")

    scenarios = {
        "01_normal_7days": generate_normal_7days(BASE_TS),
        "02_heatstroke": generate_heatstroke(BASE_TS),
        "03_fall": generate_fall(BASE_TS),
        "04_hypothermia": generate_hypothermia(BASE_TS),
        "05_fire": generate_fire(BASE_TS),
        "06_fire_cancelled": generate_fire_cancelled(BASE_TS),
        "07_emergency": generate_emergency(BASE_TS),
        "08_night_wandering": generate_night_wandering(BASE_TS),
        "09_no_return": generate_no_return(BASE_TS),
    }

    print("[Scenarios]")
    for name, data in scenarios.items():
        write_json(output_dir / f"{name}.json", data)

    print("\n[Config]")
    write_json(output_dir / "profiles.json", generate_profiles())
    write_json(output_dir / "contexts.json", generate_contexts())
    write_json(output_dir / "gateway_config.json", generate_gateway_config())

    total_readings = sum(len(d["readings"]) for d in scenarios.values())
    print(f"\nTotal: {len(scenarios)} scenarios, {total_readings:,} sensor readings generated.")


if __name__ == "__main__":
    main()
