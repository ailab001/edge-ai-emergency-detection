"""
Edge AI end-to-end simulation.

Simulates sensor data arriving in real-time and traces the full pipeline:
  Sensor → ExpertEngine + EdgePipeline → PredictionPipeline → EdgeCompactPayload

Usage:
    python -m edge_ai.datasets.simulate
    python -m edge_ai.datasets.simulate --scenario 02_heatstroke.json
    python -m edge_ai.datasets.simulate --scenario 05_fire.json --cycle 30
"""
from __future__ import annotations

import argparse
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
    TransmissionTrigger,
)
from edge_ai.expert_system import ExpertEngine
from edge_ai.pattern_matching import EdgePipeline
from edge_ai.prediction import EdgePredictionPipeline


SENSOR_CONSTRUCTORS = {
    "temp": TemperatureReading,
    "humi": HumidityReading,
    "lux": IlluminanceReading,
    "activity": ActivityReading,
    "pir": MotionReading,
    "fire": FireAlertReading,
    "emergency": EmergencyCallReading,
    "door": DoorReading,
}


def ts_fmt(ts: float, base: float) -> str:
    elapsed = ts - base
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return f"+{h:02d}:{m:02d}:{s:02d}"


def reading_summary(raw: dict) -> str:
    st = raw.get("sensor_type", "?")
    sid = raw.get("sensor_id", "?")
    if st in ("temp", "humi", "lux"):
        return f"{sid}({st}) = {raw.get('value')}{raw.get('unit', '')}"
    if st in ("activity", "pir"):
        return f"{sid}({st}) = {raw.get('value')} count={raw.get('count')}"
    if st in ("fire", "emergency"):
        return f"{sid}({st}) state={raw.get('state')}"
    if st == "door":
        return f"{sid}({st}) state={raw.get('state')} dir={raw.get('direction')}"
    return f"{sid}({st})"


def run_simulation(
    scenario_path: Path,
    profile: PersonalProfile,
    contexts: list[dict],
    cycle_interval: int = 60,
):
    with open(scenario_path, encoding="utf-8") as f:
        scenario = json.load(f)

    name = scenario["scenario"]
    readings = scenario.get("readings", [])
    mobile = scenario.get("mobile", [])

    if not readings:
        print("No readings in scenario.")
        return

    base_ts = readings[0]["ts"]

    print(f"\n{'='*80}")
    print(f"  SIMULATION: {name}")
    print(f"  {scenario.get('description', '')}")
    print(f"  Readings: {len(readings)} | Mobile: {len(mobile)} | Cycle: {cycle_interval}s")
    print(f"{'='*80}\n")

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
    prediction = EdgePredictionPipeline(home_location=profile.home_location)

    for ctx_data in contexts:
        pipeline.update_context(ExternalContext(**ctx_data))

    gps_queue = sorted(mobile, key=lambda m: m["ts"])
    gps_idx = 0

    last_cycle_ts = base_ts
    cycle_count = 0
    expert_events_all = []
    payload = None

    print(f"  [T={ts_fmt(base_ts, base_ts)}] Simulation start")
    print(f"  {'─'*76}")

    for i, raw in enumerate(readings):
        st = raw.get("sensor_type")
        constructor = SENSOR_CONSTRUCTORS.get(st)
        if constructor is None:
            continue

        reading = constructor(**raw)
        current_ts = reading.ts

        while gps_idx < len(gps_queue) and gps_queue[gps_idx]["ts"] <= current_ts:
            g = gps_queue[gps_idx]
            if "risk_level" not in g:
                gps = MobileGPS(**g)
                prediction.update_gps(gps)
                print(f"  [T={ts_fmt(g['ts'], base_ts)}] GPS  ({gps.lat:.4f}, {gps.lon:.4f})")
            gps_idx += 1

        expert_result = expert.handle_reading(reading)
        if expert_result is not None:
            expert_events_all.append(expert_result)
            status = "CANCELLED" if expert_result.cancelled else "DISPATCHED"
            print(f"  [T={ts_fmt(current_ts, base_ts)}] *** EXPERT: {expert_result.event_type} → {status} (action: {expert_result.action_taken}) ***")

        pipeline.ingest(reading)

        timeout_events = expert.check_timeouts(current_ts)
        for ev in timeout_events:
            expert_events_all.append(ev)
            print(f"  [T={ts_fmt(current_ts, base_ts)}] *** EXPERT TIMEOUT: {ev.event_type} → DISPATCHED (action: {ev.action_taken}) ***")

        if current_ts - last_cycle_ts >= cycle_interval:
            cycle_count += 1
            preprocessed = pipeline.run_cycle(current_ts)
            prediction.record(preprocessed)

            triggered = [s for s in preprocessed.cross_scenarios if s.triggered]
            grade = preprocessed.situation_grade.value

            parts = [f"grade={grade}"]
            if triggered:
                parts.append(f"scenarios=[{', '.join(s.scenario_type for s in triggered)}]")

            anomalies = [ch for ch in preprocessed.channels if ch.is_anomaly]
            if anomalies:
                parts.append(f"anomalies={len(anomalies)}")

            print(f"  [T={ts_fmt(current_ts, base_ts)}] CYCLE #{cycle_count:03d}  {' | '.join(parts)}")

            last_cycle_ts = current_ts

        elif i < 10 or st in ("fire", "emergency", "door"):
            print(f"  [T={ts_fmt(current_ts, base_ts)}] IN   {reading_summary(raw)}")

    final_ts = readings[-1]["ts"]

    final_timeout = expert.check_timeouts(final_ts + 16.0)
    for ev in final_timeout:
        expert_events_all.append(ev)
        print(f"  [T={ts_fmt(final_ts + 16, base_ts)}] *** EXPERT TIMEOUT: {ev.event_type} → DISPATCHED (action: {ev.action_taken}) ***")

    final_preprocessed = pipeline.run_cycle(final_ts + 60)
    prediction.record(final_preprocessed)

    dispatched = [ev for ev in expert_events_all if not ev.cancelled]
    if dispatched:
        trigger = TransmissionTrigger.EXPERT_EVENT
    elif final_preprocessed.situation_grade.value in ("caution", "danger"):
        trigger = TransmissionTrigger.GRADE_ESCALATION
    else:
        trigger = TransmissionTrigger.PERIODIC

    payload = prediction.build_payload(final_preprocessed, trigger, dispatched)

    print(f"\n  {'='*76}")
    print(f"  FINAL OUTPUT: EdgeCompactPayload")
    print(f"  {'='*76}")
    print(f"  subject_id:    {payload.subject_id}")
    print(f"  gateway_id:    {payload.gateway_id}")
    print(f"  trigger:       {payload.transmission_trigger.value}")
    print(f"  edge_grade:    {payload.edge_grade.value}")

    print(f"\n  [Trend Summary] window={payload.trend_summary.window_days}d, channels={len(payload.trend_summary.channels)}")
    for ch in payload.trend_summary.channels:
        print(f"    {ch.sensor_type.value:10s}  mean={ch.mean:7.2f}  std={ch.std:6.2f}  slope={ch.slope:+.6f}  [{ch.min:.1f} ~ {ch.max:.1f}]")

    print(f"\n  [Anomaly Summary] composite={payload.anomaly_summary.composite_score:.4f}")
    for a in payload.anomaly_summary.top_anomalies:
        print(f"    {a.sensor_id:5s} ({a.sensor_type.value:8s})  score={a.score:.4f}")

    print(f"\n  [Prediction]")
    print(f"    predicted_grade:      {payload.prediction.predicted_grade.value}")
    print(f"    predicted_risk_score: {payload.prediction.predicted_risk_score:.4f}")
    print(f"    horizon_minutes:      {payload.prediction.horizon_minutes}")

    if payload.gps_summary:
        g = payload.gps_summary
        print(f"\n  [GPS Summary]")
        print(f"    position:        ({g.last_lat:.4f}, {g.last_lon:.4f})")
        print(f"    home:            ({g.home_lat:.4f}, {g.home_lon:.4f})")
        print(f"    distance:        {g.distance_from_home_m:.0f}m")
        print(f"    no_return_risk:  {g.is_no_return_risk}")

    if payload.expert_events:
        print(f"\n  [Expert Events] {len(payload.expert_events)}")
        for ev in payload.expert_events:
            print(f"    {ev.event_type} | state={ev.state} | action={ev.action_taken}")

    print(f"\n  [Timestamps]")
    print(f"    window: {ts_fmt(payload.window_start, base_ts)} ~ {ts_fmt(payload.window_end, base_ts)}")
    print(f"    sent:   {ts_fmt(payload.sent_ts, base_ts)}")

    print(f"\n  {'─'*76}")
    payload_json = payload.model_dump_json(indent=2)
    print(f"  Payload size: {len(payload_json):,} bytes")

    out_path = scenario_path.parent / f"output_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload_json)
    print(f"  Saved to: {out_path}")
    print()

    logs = expert.get_judgment_logs()
    if logs:
        print(f"  [Judgment Logs for Core AI] {len(logs)}")
        for log in logs:
            print(f"    [{log.resolution:10s}] {log.event_type:10s} elapsed={log.elapsed_before_resolution:.1f}s  action={log.action_taken}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Edge AI end-to-end simulation")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario JSON file name (e.g. 02_heatstroke.json)")
    parser.add_argument("--cycle", type=int, default=60, help="Cycle interval in seconds (default: 60)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset directory path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else Path(__file__).parent

    profiles_data = json.loads((dataset_dir / "profiles.json").read_text(encoding="utf-8"))
    profile = PersonalProfile(**profiles_data[0])

    contexts_path = dataset_dir / "contexts.json"
    contexts = json.loads(contexts_path.read_text(encoding="utf-8")) if contexts_path.exists() else []

    if args.scenario:
        scenario_path = dataset_dir / args.scenario
        if not scenario_path.exists():
            print(f"Not found: {scenario_path}")
            sys.exit(1)
        ctx = contexts if "heatstroke" in args.scenario else []
        run_simulation(scenario_path, profile, ctx, args.cycle)
    else:
        for path in sorted(dataset_dir.glob("[0-9]*.json")):
            ctx = contexts if "heatstroke" in path.name else []
            run_simulation(path, profile, ctx, args.cycle)


if __name__ == "__main__":
    main()
