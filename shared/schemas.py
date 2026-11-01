from __future__ import annotations

import uuid
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SensorType(str, Enum):
    TEMP = "temp"
    HUMI = "humi"
    LUX = "lux"
    ACTIVITY = "activity"
    PIR = "pir"
    FIRE = "fire"
    EMERGENCY = "emergency"
    DOOR = "door"


class SituationGrade(str, Enum):
    WATCH = "watch"
    CAUTION = "caution"
    DANGER = "danger"


class ContextSeverity(str, Enum):
    ADVISORY = "advisory"
    WATCH = "watch"
    WARNING = "warning"
    EMERGENCY = "emergency"


class CognitiveGroup(str, Enum):
    CN = "CN"
    MCI = "MCI"
    DEM = "Dem"


class ActionType:
    MONITOR = "monitor"
    ALERT_AGENT = "alert_agent"
    DISPATCH_119 = "dispatch_119"
    CARE_CALL = "care_call"


# ---------------------------------------------------------------------------
# 1. Sensor Raw Data
# ---------------------------------------------------------------------------

class SensorBase(BaseModel):
    sensor_id: str
    device_id: str
    seq: int
    ts: float
    sensor_type: SensorType


class TemperatureReading(SensorBase):
    sensor_type: Literal[SensorType.TEMP] = SensorType.TEMP
    value: float
    unit: Literal["C"] = "C"


class HumidityReading(SensorBase):
    sensor_type: Literal[SensorType.HUMI] = SensorType.HUMI
    value: float
    unit: Literal["%"] = "%"


class IlluminanceReading(SensorBase):
    sensor_type: Literal[SensorType.LUX] = SensorType.LUX
    value: float
    unit: Literal["lx"] = "lx"


class ActivityReading(SensorBase):
    sensor_type: Literal[SensorType.ACTIVITY] = SensorType.ACTIVITY
    period_start: float
    period_end: float
    value: int
    count: int
    max: int
    min: int
    status: int


class MotionReading(SensorBase):
    sensor_type: Literal[SensorType.PIR] = SensorType.PIR
    period_start: float
    period_end: float
    value: int
    count: int
    max: int
    min: int
    status: int


class FireAlertReading(SensorBase):
    sensor_type: Literal[SensorType.FIRE] = SensorType.FIRE
    state: Literal["smoke", "normal"]
    status: int


class EmergencyCallReading(SensorBase):
    sensor_type: Literal[SensorType.EMERGENCY] = SensorType.EMERGENCY
    state: Literal["red", "normal"]
    status: int


class DoorReading(SensorBase):
    sensor_type: Literal[SensorType.DOOR] = SensorType.DOOR
    state: Literal["open", "normal"]
    direction: Literal["in", "out", "-"]
    status: int


SensorReading = Annotated[
    Union[
        TemperatureReading,
        HumidityReading,
        IlluminanceReading,
        ActivityReading,
        MotionReading,
        FireAlertReading,
        EmergencyCallReading,
        DoorReading,
    ],
    Field(discriminator="sensor_type"),
]


class MobileRiskAssessment(BaseModel):
    subject_id: str
    risk_level: str
    fall: bool
    no_return: bool
    usage_anomaly: bool
    long_inactivity: bool
    ts: float
    device_id: str


class MobileGPS(BaseModel):
    subject_id: str
    lat: float
    lon: float
    ts: float
    device_id: str


# ---------------------------------------------------------------------------
# Common / Config
# ---------------------------------------------------------------------------

class HomeLocation(BaseModel):
    lat: float
    lon: float
    address: Optional[str] = None


class GatewayConfig(BaseModel):
    gateway_id: str
    subject_id: str
    device_mappings: dict[str, SensorType]
    home_location: Optional[HomeLocation] = None


class ThresholdAdjustment(BaseModel):
    target_sensor_type: SensorType
    target_scenario: str
    parameter_name: str
    original_value: float
    adjusted_value: float
    reason: str


class ExternalContext(BaseModel):
    context_id: str
    context_type: Literal["weather_alert", "disaster_alert", "custom"]
    severity: ContextSeverity
    region_code: str
    message: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    effective_from: float
    effective_until: float
    threshold_adjustments: list[ThresholdAdjustment] = Field(default_factory=list)
    received_ts: float


class EmergencyContact(BaseModel):
    name: str
    phone: str
    relationship: str


class PersonalProfile(BaseModel):
    subject_id: str
    age: int
    cognitive_group: CognitiveGroup
    cognitive_group_updated_at: Optional[float] = None
    mobility_impaired: bool = False
    sleep_disorder: bool = False
    medications: list[str] = Field(default_factory=list)
    home_location: HomeLocation
    living_alone: bool = True
    emergency_contacts: list[EmergencyContact] = Field(default_factory=list)
    custom_thresholds: Optional[dict[str, float]] = None
    enrollment_date: float
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Edge AI Preprocessed Data
# ---------------------------------------------------------------------------

class ProcessedChannel(BaseModel):
    sensor_id: str
    sensor_type: SensorType
    raw_value: float
    kalman_value: float
    ewma_value: float
    anomaly_score: float = Field(ge=-1.0, le=1.0)
    is_anomaly: bool
    ts: float


class CrossScenarioResult(BaseModel):
    scenario_type: Literal[
        "heatstroke", "fall", "hypothermia", "wandering", "no_return"
    ]
    triggered: bool
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_sensors: list[str]


class ContextAdjustmentRecord(BaseModel):
    context_id: str
    adjustments: list[ThresholdAdjustment]
    applied_ts: float


class EdgePreprocessedData(BaseModel):
    subject_id: str
    gateway_id: str
    channels: list[ProcessedChannel]
    cross_scenarios: list[CrossScenarioResult] = Field(default_factory=list)
    situation_grade: SituationGrade
    grade_reason: str
    cycle_ts: float
    context_adjustments: list[ContextAdjustmentRecord] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# 3. Edge -> Core Compact Payload
# ---------------------------------------------------------------------------

class ChannelTrend(BaseModel):
    sensor_type: SensorType
    mean: float
    std: float
    slope: float
    min: float
    max: float


class TrendSummary(BaseModel):
    window_days: int = 7
    channels: list[ChannelTrend]


class AnomalyRank(BaseModel):
    sensor_type: SensorType
    score: float
    sensor_id: str


class AnomalySummary(BaseModel):
    composite_score: float = Field(ge=-1.0, le=1.0)
    top_anomalies: list[AnomalyRank]


class EdgePrediction(BaseModel):
    predicted_grade: SituationGrade
    predicted_risk_score: float = Field(ge=0.0, le=1.0)
    horizon_minutes: int


class GPSSummary(BaseModel):
    last_lat: float
    last_lon: float
    home_lat: float
    home_lon: float
    distance_from_home_m: float
    is_no_return_risk: bool


class ExpertEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: Literal["fire", "emergency"]
    state: str
    triggered_ts: float
    cancelled: bool = False
    action_taken: str


class TransmissionTrigger(str, Enum):
    PERIODIC = "periodic"
    GRADE_ESCALATION = "grade_escalation"
    EXPERT_EVENT = "expert_event"
    SCHEDULED = "scheduled"


class EdgeCompactPayload(BaseModel):
    subject_id: str
    gateway_id: str
    window_start: float
    window_end: float
    transmission_trigger: TransmissionTrigger
    edge_grade: SituationGrade
    trend_summary: TrendSummary
    anomaly_summary: AnomalySummary
    prediction: EdgePrediction
    gps_summary: Optional[GPSSummary] = None
    expert_events: list[ExpertEvent] = Field(default_factory=list)
    sent_ts: float


# ---------------------------------------------------------------------------
# 4. Core AI Inference Result
# ---------------------------------------------------------------------------

class EmbeddingSimilarityOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    matched_case_id: Optional[str] = None


class TransformerOutput(BaseModel):
    predicted_risk: float = Field(ge=0.0, le=1.0)
    anomaly_segments: list[tuple[float, float]] = Field(default_factory=list)


class TimesFMOutput(BaseModel):
    risk_prob_24h: float = Field(ge=0.0, le=1.0)
    trend_forecast: list[float] = Field(default_factory=list)


class ModelOutputs(BaseModel):
    embedding_similarity: Optional[EmbeddingSimilarityOutput] = None
    transformer_prediction: Optional[TransformerOutput] = None
    timesfm_prediction: Optional[TimesFMOutput] = None


class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    case_summary: str
    outcome: str


class EvidenceRef(BaseModel):
    data_type: Literal["sensor_raw", "edge_preprocessed", "stt_transcript"]
    ref_id: str
    time_range: tuple[float, float]


class CoreInferenceResult(BaseModel):
    subject_id: str
    inference_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    triggered_by: Literal["edge_escalation", "expert_event", "scheduled"]
    risk_level: float = Field(ge=0.0, le=1.0)
    prediction_grade: SituationGrade
    confidence: float = Field(ge=0.0, le=1.0)
    model_outputs: ModelOutputs
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    action_type: str = ActionType.MONITOR
    suppressed_by_expert: bool = False
    suppression_ref: Optional[str] = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    edge_payload_ref: Optional[str] = None
    inferred_ts: float
