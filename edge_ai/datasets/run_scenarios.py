"""
Edge AI scenario runner for local testing.

Loads JSON test datasets and runs them through ExpertEngine + EdgePipeline,
printing results for each scenario.

Usage:
    python -m edge_ai.datasets.run_scenarios
    python -m edge_ai.datasets.run_scenarios --dataset ./edge_ai/datasets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from shared.schemas import (
    ActivityReading,
    DoorReading,
    EmergencyCallReading,
    ExternalContext,
    FireAlertReading,
    GatewayConfig,
    HomeLocation,
    HumidityReading,
    IlluminanceReading,
    MobileGPS,
    MotionReading,
    PersonalProfile,
    SensorType,
    TemperatureReading,
)
from edge_ai.expert_system import ExpertEngine
from edge_ai.pattern_matching import EdgePipeline


SENSOR_CONSTRUCTORS = {
    "temp": lambda d: TemperatureReading(**d),
    "humi": lambda d: HumidityReading(**d),
    "lux": lambda d: IlluminanceReading(**d),
    "activity": lambda d: ActivityReading(**d),
    "pir": lambda d: MotionReading(**d),
    "fire": lambda d: FireAlertReading(**d),
    "emergency": lambda d: EmergencyCallReading(**d),
    "door": lambda d: DoorReading(**d),
}


def parse_reading(raw: dict):
    sensor_type = raw.get("sensor_type")
    constructor = SENSOR_CONSTRUCTORS.get(sensor_type)
    if constructor is None:
        return None
    return constructor(raw)


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_scenario(scenario_data: dict, profile: PersonalProfile, contexts: list[dict]):
    name = scenario_data["scenario"]
    description = scenario_data["description"]
    expected = scenario_data.get("expected_grade", "unknown")
    expected_action = scenario_data.get("expected_action")

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {name}")
    print(f"  {description}")
    print(f"  Expected: {expected}" + (f" / action: {expected_action}" if expected_action else ""))
    print(f"{'='*70}")

    gw_config = GatewayConfig(
        gateway_id="gw_001",
        subject_id=profile.subject_id,
        device_mappings={
            "T01": SensorType.TEMP, "H01": SensorType.HUMI, "L01": SensorType.LUX,
            "A01": SensorType.ACTIVITY, "P01": SensorType.PIR,
            "F01": SensorType.FIRE, "E01": SensorType.EMERGENCY, "D01": SensorType.DOOR,
        },
        home_location=profile.home_location,
    )

    expert = ExpertEngine(subject_id=profile.subject_id)
    pipeline = EdgePipeline(gw_config, profile)

    for ctx_data in contexts:
        ctx = ExternalContext(**ctx_data)
        pipeline.update_context(ctx)

    readings = scenario_data.get("readings", [])
    expert_events = []
    last_ts = 0.0

    for raw in readings:
        reading = parse_reading(raw)
        if reading is None:
            continue

        last_ts = max(last_ts, reading.ts)
        expert_result = expert.handle_reading(reading)
        if expert_result is not None:
            expert_events.append(expert_result)

        pipeline.ingest(reading)

    timeout_events = expert.check_timeouts(last_ts + 16.0)
    expert_events.extend(timeout_events)

    cycle_ts = last_ts + 60.0
    preprocessed = pipeline.run_cycle(cycle_ts)

    print(f"\n  [Expert System]")
    if expert_events:
        for ev in expert_events:
            status = "CANCELLED" if ev.cancelled else "DISPATCHED"
            print(f"    {ev.event_type} -> {status} (action: {ev.action_taken})")
    else:
        print(f"    No expert events triggered")

    logs = expert.get_judgment_logs()
    if logs:
        print(f"    Judgment logs: {len(logs)}")
        for log in logs:
            print(f"      [{log.resolution}] {log.event_type} elapsed={log.elapsed_before_resolution:.1f}s")

    print(f"\n  [Pattern Matching Pipeline]")
    print(f"    Situation grade: {preprocessed.situation_grade.value}")
    print(f"    Reason: {preprocessed.grade_reason}")
    print(f"    Channels processed: {len(preprocessed.channels)}")

    if preprocessed.cross_scenarios:
        print(f"    Cross-scenarios:")
        for cs in preprocessed.cross_scenarios:
            flag = "TRIGGERED" if cs.triggered else "not triggered"
            print(f"      {cs.scenario_type}: {flag} (confidence={cs.confidence:.2f})")

    if preprocessed.context_adjustments:
        print(f"    Context adjustments applied: {len(preprocessed.context_adjustments)}")

    anomalies = [ch for ch in preprocessed.channels if ch.is_anomaly]
    if anomalies:
        print(f"    Anomalous channels: {len(anomalies)}")
        for ch in sorted(anomalies, key=lambda c: c.anomaly_score, reverse=True)[:3]:
            print(f"      {ch.sensor_id} ({ch.sensor_type.value}): score={ch.anomaly_score:.3f}")

    mobile = scenario_data.get("mobile", [])
    if mobile:
        gps_data = [m for m in mobile if "lat" in m and "lon" in m and "risk_level" not in m]
        risk_data = [m for m in mobile if "risk_level" in m]
        print(f"\n  [Mobile Data]")
        if gps_data:
            last_gps = gps_data[-1]
            print(f"    Last GPS: ({last_gps['lat']:.4f}, {last_gps['lon']:.4f})")
        if risk_data:
            for r in risk_data:
                print(f"    Risk assessment: level={r['risk_level']} fall={r['fall']} no_return={r['no_return']}")

    print(f"\n  Total readings: {len(readings)}")

    dispatched = [ev for ev in expert_events if not ev.cancelled]
    pipeline_grade = preprocessed.situation_grade.value

    if dispatched:
        combined_grade = "danger"
        combined_reason = f"expert:{dispatched[0].event_type}"
    else:
        combined_grade = pipeline_grade
        combined_reason = preprocessed.grade_reason

    print(f"\n  [Combined Judgment]")
    print(f"    Expert events (dispatched): {len(dispatched)}")
    print(f"    Pipeline grade: {pipeline_grade}")
    print(f"    -> Final grade: {combined_grade} ({combined_reason})")

    match = "PASS" if expected in ("unknown", combined_grade) or combined_grade in expected else "CHECK"
    print(f"  Result: [{match}]")
    return combined_grade, match


def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    profiles_path = dataset_dir / "profiles.json"
    contexts_path = dataset_dir / "contexts.json"

    if not profiles_path.exists():
        print(f"profiles.json not found in {dataset_dir}")
        print("Run: python -m edge_ai.datasets.generator")
        sys.exit(1)

    profiles_data = load_json(profiles_path)
    profile = PersonalProfile(**profiles_data[0])

    contexts_data = load_json(contexts_path) if contexts_path.exists() else []

    scenario_files = sorted(dataset_dir.glob("[0-9]*.json"))
    if not scenario_files:
        print(f"No scenario files found in {dataset_dir}")
        sys.exit(1)

    print(f"Profile: {profile.subject_id} (age={profile.age}, group={profile.cognitive_group.value})")
    print(f"Scenarios: {len(scenario_files)}")

    results = []
    for path in scenario_files:
        data = load_json(path)
        ctx = contexts_data if "heatstroke" in data["scenario"] else []
        grade, match = run_scenario(data, profile, ctx)
        results.append((data["scenario"], grade, match))

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for name, grade, match in results:
        print(f"  [{match:5s}] {name:25s} -> {grade}")

    checks = sum(1 for _, _, m in results if m == "CHECK")
    if checks:
        print(f"\n  {checks} scenario(s) need manual verification.")
    else:
        print(f"\n  All scenarios passed expected grade checks.")


if __name__ == "__main__":
    main()
