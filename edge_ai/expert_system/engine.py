from __future__ import annotations

import uuid
from typing import Optional, Union

from shared.schemas import (
    ActionType,
    ExpertEvent,
    FireAlertReading,
    EmergencyCallReading,
    SensorReading,
    SensorType,
)
from edge_ai.expert_system.config import (
    AnnounceRequest,
    DispatchRequest,
    ExpertCallback,
    RuleConfig,
    DEFAULT_RULES,
)
from edge_ai.expert_system.models import EventStatus, JudgmentLog, PendingEvent


class ExpertEngine:
    def __init__(
        self,
        rules: dict[SensorType, RuleConfig] | None = None,
        callback: ExpertCallback | None = None,
        subject_id: str = "",
    ):
        self._rules = rules or DEFAULT_RULES
        self._callback = callback
        self._subject_id = subject_id
        self._pending: dict[str, PendingEvent] = {}
        self._cooldowns: dict[str, float] = {}
        self._log_buffer: list[JudgmentLog] = []

    def handle_reading(self, reading: SensorReading) -> ExpertEvent | None:
        rule = self._rules.get(reading.sensor_type)
        if rule is None:
            return None

        state = self._extract_state(reading)
        if state is None:
            return None

        if state == rule.cancel_state:
            return self._handle_cancel(reading, rule)

        if state == rule.trigger_state:
            return self._handle_trigger(reading, rule)

        return None

    def check_timeouts(self, now: float) -> list[ExpertEvent]:
        events: list[ExpertEvent] = []

        expired_pending = [
            sid for sid, pe in self._pending.items()
            if pe.status == EventStatus.PENDING and now >= pe.deadline_ts
        ]
        for sensor_id in expired_pending:
            pe = self._pending.pop(sensor_id)
            pe.status = EventStatus.DISPATCHED
            pe.resolved_ts = pe.deadline_ts
            self._cooldowns[sensor_id] = pe.deadline_ts + pe.rule.cooldown_sec
            event = self._to_expert_event(pe, cancelled=False)
            self._emit_log(pe)
            if self._callback:
                self._callback.on_dispatch(DispatchRequest(
                    sensor_id=sensor_id,
                    event_type=pe.rule.event_type,
                    event_id=pe.event_id,
                ))
            events.append(event)

        expired_cooldowns = [
            sid for sid, deadline in self._cooldowns.items()
            if now >= deadline
        ]
        for sensor_id in expired_cooldowns:
            del self._cooldowns[sensor_id]

        return events

    def get_judgment_logs(self) -> list[JudgmentLog]:
        logs = list(self._log_buffer)
        self._log_buffer.clear()
        return logs

    def _handle_trigger(
        self, reading: SensorReading, rule: RuleConfig
    ) -> ExpertEvent | None:
        sensor_id = reading.sensor_id
        if sensor_id in self._pending:
            return None
        if sensor_id in self._cooldowns:
            return None

        event_id = uuid.uuid4().hex[:12]
        pe = PendingEvent(
            event_id=event_id,
            sensor_id=sensor_id,
            rule=rule,
            triggered_ts=reading.ts,
            deadline_ts=reading.ts + rule.cancel_window_sec,
            trigger_reading=reading.model_dump(),
        )
        self._pending[sensor_id] = pe

        if self._callback:
            self._callback.on_announce(AnnounceRequest(
                sensor_id=sensor_id,
                event_type=rule.event_type,
                announce_count=rule.announce_count,
                announce_interval_sec=rule.announce_interval_sec,
                cancel_window_sec=rule.cancel_window_sec,
            ))

        return None

    def _handle_cancel(
        self, reading: SensorReading, rule: RuleConfig
    ) -> ExpertEvent | None:
        sensor_id = reading.sensor_id
        pe = self._pending.pop(sensor_id, None)
        if pe is None:
            return None

        pe.status = EventStatus.CANCELLED
        pe.cancel_reading = reading.model_dump()
        pe.resolved_ts = reading.ts
        self._emit_log(pe)
        return self._to_expert_event(pe, cancelled=True)

    def _to_expert_event(self, pe: PendingEvent, cancelled: bool) -> ExpertEvent:
        return ExpertEvent(
            event_id=pe.event_id,
            event_type=pe.rule.event_type,
            state=pe.trigger_reading.get("state", ""),
            triggered_ts=pe.triggered_ts,
            cancelled=cancelled,
            action_taken=ActionType.MONITOR if cancelled else pe.rule.action_on_confirm,
        )

    def _emit_log(self, pe: PendingEvent) -> None:
        resolved_ts = pe.resolved_ts or pe.deadline_ts
        self._log_buffer.append(JudgmentLog(
            event_id=pe.event_id,
            subject_id=self._subject_id,
            sensor_type=pe.rule.sensor_type,
            event_type=pe.rule.event_type,
            trigger_reading=pe.trigger_reading,
            cancel_reading=pe.cancel_reading,
            triggered_ts=pe.triggered_ts,
            resolved_ts=resolved_ts,
            resolution="cancelled" if pe.status == EventStatus.CANCELLED else "dispatched",
            cancel_window_sec=pe.rule.cancel_window_sec,
            elapsed_before_resolution=resolved_ts - pe.triggered_ts,
            action_taken=ActionType.MONITOR if pe.status == EventStatus.CANCELLED else pe.rule.action_on_confirm,
        ))

    @staticmethod
    def _extract_state(
        reading: SensorReading,
    ) -> Optional[str]:
        if isinstance(reading, (FireAlertReading, EmergencyCallReading)):
            return reading.state
        return None
