import pytest

from shared.schemas import ActionType, SensorType
from edge_ai.expert_system import ExpertEngine, AnnounceRequest, DispatchRequest
from edge_ai.tests.conftest import (
    BASE_TS,
    make_fire,
    make_emergency,
    make_temp,
)


class MockCallback:
    def __init__(self):
        self.announces: list[AnnounceRequest] = []
        self.dispatches: list[DispatchRequest] = []

    def on_announce(self, request: AnnounceRequest) -> None:
        self.announces.append(request)

    def on_dispatch(self, request: DispatchRequest) -> None:
        self.dispatches.append(request)


class TestFireDetection:
    def test_smoke_triggers_pending(self):
        engine = ExpertEngine(subject_id="U001")
        result = engine.handle_reading(make_fire(BASE_TS, "smoke"))
        assert result is None

    def test_smoke_timeout_dispatches_119(self):
        cb = MockCallback()
        engine = ExpertEngine(callback=cb, subject_id="U001")

        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        assert len(cb.announces) == 1
        assert cb.announces[0].cancel_window_sec == 15.0

        events = engine.check_timeouts(BASE_TS + 16.0)
        assert len(events) == 1
        assert events[0].event_type == "fire"
        assert events[0].cancelled is False
        assert events[0].action_taken == ActionType.DISPATCH_119
        assert len(cb.dispatches) == 1

    def test_smoke_cancel_within_window(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))

        result = engine.handle_reading(make_fire(BASE_TS + 10.0, "normal"))
        assert result is not None
        assert result.cancelled is True
        assert result.action_taken == ActionType.MONITOR

    def test_cancel_generates_judgment_log(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.handle_reading(make_fire(BASE_TS + 10.0, "normal"))

        logs = engine.get_judgment_logs()
        assert len(logs) == 1
        assert logs[0].resolution == "cancelled"
        assert logs[0].elapsed_before_resolution == 10.0
        assert logs[0].subject_id == "U001"

    def test_cooldown_blocks_retrigger(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.check_timeouts(BASE_TS + 16.0)

        engine.handle_reading(make_fire(BASE_TS + 20.0, "smoke"))
        events = engine.check_timeouts(BASE_TS + 36.0)
        assert len(events) == 0

    def test_cooldown_expires(self):
        cb = MockCallback()
        engine = ExpertEngine(callback=cb, subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.check_timeouts(BASE_TS + 16.0)

        engine.check_timeouts(BASE_TS + 400.0)

        engine.handle_reading(make_fire(BASE_TS + 401.0, "smoke"))
        events = engine.check_timeouts(BASE_TS + 417.0)
        assert len(events) == 1
        assert len(cb.dispatches) == 2


class TestEmergencyCall:
    def test_emergency_timeout_dispatches(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_emergency(BASE_TS, "red"))
        events = engine.check_timeouts(BASE_TS + 16.0)
        assert len(events) == 1
        assert events[0].event_type == "emergency"
        assert events[0].action_taken == ActionType.DISPATCH_119

    def test_emergency_cancel(self):
        engine = ExpertEngine(subject_id="U001")
        engine.handle_reading(make_emergency(BASE_TS, "red"))
        result = engine.handle_reading(make_emergency(BASE_TS + 5.0, "normal"))
        assert result is not None
        assert result.cancelled is True


class TestIrrelevantSensors:
    def test_temp_reading_ignored(self):
        engine = ExpertEngine(subject_id="U001")
        result = engine.handle_reading(make_temp(BASE_TS, 28.0))
        assert result is None

    def test_no_timeout_without_pending(self):
        engine = ExpertEngine(subject_id="U001")
        events = engine.check_timeouts(BASE_TS + 100.0)
        assert events == []


class TestDuplicateTrigger:
    def test_duplicate_smoke_ignored(self):
        cb = MockCallback()
        engine = ExpertEngine(callback=cb, subject_id="U001")
        engine.handle_reading(make_fire(BASE_TS, "smoke"))
        engine.handle_reading(make_fire(BASE_TS + 2.0, "smoke"))
        assert len(cb.announces) == 1
