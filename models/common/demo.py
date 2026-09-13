"""Explicit synthetic signals for UI integration, never model predictions."""

import math
import statistics
from typing import Any

from models.common.contracts import ContractError, InsufficientHistory, timestamp


def numeric(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite.")
    return result


def observe(code: str, description: str) -> dict[str, str]:
    return {"code": code, "description": description, "kind": "observation"}


def demo_prediction(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    reasons = [observe("SYNTHETIC_DEMO", "Illustrative rule signal; no trained model is loaded.")]
    if kind == "onboarding":
        application = payload["application"]
        required = ("credit_risk_score", "identity_mismatch", "device_shared_count")
        missing = [key for key in required if key not in application]
        if missing:
            raise ContractError("Missing demo application fields: " + ", ".join(missing))
        base = numeric(application["credit_risk_score"], "credit_risk_score")
        shared = numeric(application["device_shared_count"], "device_shared_count")
        mismatch = application["identity_mismatch"]
        if not isinstance(mismatch, bool):
            raise ContractError("identity_mismatch must be a boolean.")
        if not 0 <= base <= 400 or shared < 0:
            raise ContractError("credit_risk_score must be 0–400; device_shared_count must be nonnegative.")
        risk = min(0.99, base / 300 + 0.24 * int(mismatch) + min(0.20, max(0, shared - 1) * 0.04))
        if mismatch:
            reasons.append(observe("IDENTITY_MISMATCH", "Synthetic identity fields do not match."))
        if shared > 1:
            reasons.append(observe("SHARED_DEVICE", f"Synthetic device is linked to {shared:g} applications."))
        features = {key: application[key] for key in required}
    else:
        events = payload["events"]
        if kind == "sequence" and len(events) < 4:
            raise InsufficientHistory("Demo sequence requires at least 4 observed events.")
        if not events:
            raise InsufficientHistory("A transaction event is required.")
        outgoing = [event for event in events if event.get("type", "").lower() not in {"deposit", "cash_in"}]
        amounts = [abs(numeric(event.get("amount", 0), "amount")) for event in outgoing]
        baseline = statistics.median(amounts[:3]) if amounts else 1
        baseline = max(1.0, baseline)
        last_amount = abs(numeric(events[-1].get("amount", 0), "amount"))
        last_outgoing = events[-1].get("type", "").lower() not in {"deposit", "cash_in"}
        ratio = last_amount / baseline if last_outgoing else 0.0
        spike_count = sum(amount > baseline * 8 for amount in amounts[3:])
        cutoff = timestamp(payload["as_of"])
        recent_count = sum(0 <= (cutoff - timestamp(event["event_time"])).total_seconds() <= 300 for event in outgoing)
        if kind == "transaction":
            risk = min(0.99, 0.12 + min(0.68, max(0, ratio - 2) * 0.045) + min(0.19, spike_count * 0.10))
        else:
            risk = min(0.99, 0.14 + min(0.74, spike_count * 0.37) + min(0.11, max(0, recent_count - 3) * 0.055))
        if ratio > 8:
            reasons.append(observe("AMOUNT_SPIKE", f"Outgoing amount is {ratio:.1f}× the initial three-event median."))
        if spike_count:
            reasons.append(observe("REPEATED_OUTFLOW", f"{spike_count} outgoing event(s) exceed 8× the initial median."))
        if recent_count > 3:
            reasons.append(observe("VELOCITY_INCREASE", f"{recent_count} outgoing events occurred in the last five minutes."))
        features = {
            "event_count": len(events), "baseline_outgoing_median": baseline,
            "last_amount": last_amount, "last_outgoing_ratio": ratio,
            "spike_count": spike_count, "outgoing_count_5m": recent_count,
        }
    return {"risk_score": round(risk, 6), "raw_score": round(risk, 6), "reason_codes": reasons, "features": features}
