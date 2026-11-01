from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from shared.schemas import ActionType, SensorType


class RuleConfig(BaseModel):
    sensor_type: SensorType
    event_type: Literal["fire", "emergency"]
    trigger_state: str
    cancel_state: str
    cancel_window_sec: float
    announce_count: int
    announce_interval_sec: float
    action_on_confirm: str
    cooldown_sec: float


class AnnounceRequest(BaseModel):
    sensor_id: str
    event_type: str
    announce_count: int
    announce_interval_sec: float
    cancel_window_sec: float


class DispatchRequest(BaseModel):
    sensor_id: str
    event_type: str
    event_id: str


class ExpertCallback(Protocol):
    def on_announce(self, request: AnnounceRequest) -> None: ...
    def on_dispatch(self, request: DispatchRequest) -> None: ...


DEFAULT_RULES: dict[SensorType, RuleConfig] = {
    SensorType.FIRE: RuleConfig(
        sensor_type=SensorType.FIRE,
        event_type="fire",
        trigger_state="smoke",
        cancel_state="normal",
        cancel_window_sec=15.0,
        announce_count=2,
        announce_interval_sec=7.5,
        action_on_confirm=ActionType.DISPATCH_119,
        cooldown_sec=300.0,
    ),
    SensorType.EMERGENCY: RuleConfig(
        sensor_type=SensorType.EMERGENCY,
        event_type="emergency",
        trigger_state="red",
        cancel_state="normal",
        cancel_window_sec=15.0,
        announce_count=2,
        announce_interval_sec=7.5,
        action_on_confirm=ActionType.DISPATCH_119,
        cooldown_sec=300.0,
    ),
}
