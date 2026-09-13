"""Causal transaction features matching the delivered LightGBM notebook."""

import math

from models.common.contracts import ContractError, timestamp


RAW_FIELDS = ["card_id","device_id","ip_address","merchant_id","merchant_category","merchant_country","merchant_city","merchant_latitude","merchant_longitude"]


def raw(event, name):
    if name in event:
        return event[name]
    value = event.get("raw_payload", {}).get(name)
    if value is None:
        raise ContractError(f"Transaction {event.get('id', '<unknown>')} is missing {name}.")
    return value


def haversine(lat1, lon1, lat2, lon2):
    values = [math.radians(float(value)) for value in (lat1, lon1, lat2, lon2)]
    lat1, lon1, lat2, lon2 = values
    value = math.sin((lat2-lat1)/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(value))


def build_features(payload):
    events = payload.get("events", [])
    if not events:
        raise ContractError("Transaction scoring requires at least one event.")
    rows = []
    seen_devices = set()
    amounts = []
    for event in events:
        values = {name: raw(event, name) for name in RAW_FIELDS}
        event_time = timestamp(event["event_time"])
        previous = rows[-1] if rows else None
        delta = max(0.0, (event_time - previous["_timestamp"]).total_seconds()) if previous else 0.0
        distance = haversine(values["merchant_latitude"], values["merchant_longitude"], previous["_latitude"], previous["_longitude"]) if previous else 0.0
        amount = float(event["amount"])
        prior_average = sum(amounts) / len(amounts) if amounts else amount
        one_hour = [item for item in rows if (event_time-item["_timestamp"]).total_seconds() < 3600]
        one_day = [item for item in rows if (event_time-item["_timestamp"]).total_seconds() < 86400]
        transaction_type = str(event.get("raw_payload", {}).get("transaction_type", event["type"]))
        row = {
            "merchant_category": str(values["merchant_category"]), "merchant_country": str(values["merchant_country"]),
            "merchant_city": str(values["merchant_city"]), "transaction_type": transaction_type, "amount": amount,
            "hour_of_day": event_time.hour, "day_of_week": event_time.weekday(), "is_night_time": int(1 <= event_time.hour <= 5),
            "time_since_last_tx": delta, "geovelocity_kmh": distance / ((delta + 1) / 3600),
            "is_new_device": int(values["device_id"] not in seen_devices), "tx_count_1h": float(len(one_hour)+1),
            "tx_count_24h": float(len(one_day)+1), "tx_amount_1h": float(sum(item["amount"] for item in one_hour)+amount),
            "tx_amount_24h": float(sum(item["amount"] for item in one_day)+amount), "amount_to_avg_ratio": amount/(prior_average+0.01),
            "device_tx_count_24h": 0.0,
            "_timestamp": event_time, "_latitude": float(values["merchant_latitude"]), "_longitude": float(values["merchant_longitude"]),
            "_device_id": values["device_id"]
        }
        rows.append(row)
        amounts.append(amount)
        seen_devices.add(values["device_id"])

    latest = rows[-1]
    portfolio = payload.get("context", {}).get("portfolio_events", events)
    latest_time = latest["_timestamp"]
    device_count = 0
    for event in portfolio:
        try:
            same_device = raw(event, "device_id") == latest["_device_id"]
            age = (latest_time - timestamp(event["event_time"])).total_seconds()
            device_count += int(same_device and 0 <= age < 86400)
        except ContractError:
            continue
    latest["device_tx_count_24h"] = float(device_count or 1)
    return {key:value for key,value in latest.items() if not key.startswith("_")}


def reason_codes(payload, snapshot, raw_score):
    row = dict(zip(snapshot["feature_order"], snapshot["raw_vector"][-1]))
    reasons = []
    if row["amount_to_avg_ratio"] >= 3:
        reasons.append({"code":"AMOUNT_SPIKE","description":"Amount is at least three times the prior account average.","kind":"observation"})
    if row["tx_count_1h"] >= 3:
        reasons.append({"code":"TRANSACTION_BURST","description":"At least three transactions occurred within one hour.","kind":"observation"})
    if row["geovelocity_kmh"] >= 500:
        reasons.append({"code":"IMPOSSIBLE_TRAVEL","description":"Merchant locations imply unusually high travel velocity.","kind":"observation"})
    if row["is_new_device"]:
        reasons.append({"code":"NEW_DEVICE","description":"Transaction uses a device not previously observed on the account.","kind":"observation"})
    return reasons
