"""Canonical integration boundary, independent of notebook feature layouts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ApplicationInput(Contract):
    application_id: str = Field(min_length=1, max_length=120)
    account_id: str = Field(min_length=1, max_length=120)
    alias: str = Field(min_length=1, max_length=100)
    event_time: datetime
    raw_payload: dict[str, Any]
    labels: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("event_time must include a timezone")
        return value


class EventInput(Contract):
    event_id: str = Field(min_length=1, max_length=120)
    account_id: str = Field(min_length=1, max_length=120)
    event_time: datetime
    type: Literal["deposit", "purchase", "transfer", "cash_out", "withdrawal", "payment"]
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", pattern="^[A-Z]{3}$")
    counterparty_id: str | None = Field(default=None, max_length=120)
    description: str = Field(default="", max_length=500)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("event_time must include a timezone")
        return value


class ThresholdInput(Contract):
    threshold: float = Field(ge=0, le=1)
    onboarding_threshold: float | None = Field(default=None, ge=0, le=1)


class SpeedInput(Contract):
    speed: Literal[1, 5, 20]


class ReviewInput(Contract):
    status: Literal["escalated", "dismissed"]
    notes: str = Field(min_length=1, max_length=2000)


class Reason(Contract):
    code: str
    description: str
    kind: Literal["observation", "model"]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    model: Literal["onboarding", "transaction", "sequence"]
    model_version: str = "unavailable"
    risk_score: float | None = Field(default=None, ge=0, le=1)
    raw_score: float | None = None
    status: Literal["ok", "demo", "not_ready", "insufficient_history", "error"]
    reason_codes: list[Reason] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    preprocessing_version: str | None = None
    contract_hash: str | None = None
    latency_ms: float = 0
