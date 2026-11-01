from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel

from shared.schemas import SensorType
from edge_ai.expert_system.config import RuleConfig


class EventStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"


class PendingEvent(BaseModel):
    event_id: str
    sensor_id: str
    rule: RuleConfig
    triggered_ts: float
    deadline_ts: float
    status: EventStatus = EventStatus.PENDING
    trigger_reading: dict[str, Any] = {}
    cancel_reading: Optional[dict[str, Any]] = None
    resolved_ts: Optional[float] = None


class JudgmentLog(BaseModel):
    event_id: str
    subject_id: str
    sensor_type: SensorType
    event_type: str
    trigger_reading: dict[str, Any]
    cancel_reading: Optional[dict[str, Any]] = None
    triggered_ts: float
    resolved_ts: float
    resolution: Literal["dispatched", "cancelled"]
    cancel_window_sec: float
    elapsed_before_resolution: float
    action_taken: str
