"""Validated HTTP types and deterministic, leakage-safe model payloads."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


LABEL_KEYS = {
    "label", "labels", "target", "y", "fraud", "fraud_bool", "isfraud",
    "is_fraud", "isflaggedfraud", "ground_truth", "evaluation", "outcome",
}


def without_labels(value: Any, extra_keys: set[str] | None = None) -> Any:
    excluded = LABEL_KEYS | (extra_keys or set())
    if isinstance(value, dict):
        return {
            key: without_labels(item, excluded)
            for key, item in value.items()
            if key.lower() not in excluded
        }
    if isinstance(value, list):
        return [without_labels(item, excluded) for item in value]
    return value


def timestamp(value: str | datetime) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone.")
    return result.astimezone(timezone.utc)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=200)
    as_of: datetime
    application: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list, max_length=10000)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value: datetime) -> datetime:
        return timestamp(value)

    @field_validator("events")
    @classmethod
    def valid_event_times(cls, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in events:
            if "event_time" not in event:
                raise ValueError("Each event requires event_time.")
            timestamp(event["event_time"])
        return events

    def safe_payload(self, extra_labels: set[str] | None = None) -> dict[str, Any]:
        payload = without_labels(self.model_dump(mode="json"), extra_labels)
        cutoff = timestamp(payload["as_of"])
        payload["events"] = sorted(
            (event for event in payload["events"] if timestamp(event["event_time"]) <= cutoff),
            key=lambda event: (
                timestamp(event["event_time"]),
                event.get("sequence_number", 0),
                str(event.get("id", event.get("event_id", ""))),
            ),
        )
        return payload


class ReasonCode(BaseModel):
    code: str
    description: str
    kind: Literal["observation", "model"] = "observation"


class PredictionResponse(BaseModel):
    model: str
    model_version: str
    risk_score: float | None = Field(default=None, ge=0, le=1)
    raw_score: float | None = None
    status: Literal["ok", "demo", "not_ready", "insufficient_history", "error"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    preprocessing_version: str
    contract_hash: str
    latency_ms: float


class ContractError(ValueError):
    """Input or exported model metadata does not satisfy the contract."""


class InsufficientHistory(ContractError):
    """The sequence cannot be formed without fabricating history."""
