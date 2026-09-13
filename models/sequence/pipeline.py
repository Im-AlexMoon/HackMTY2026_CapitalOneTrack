"""Causal sequence features matching the delivered LSTM autoencoder script."""

import math

from models.common.contracts import ContractError, timestamp

CATEGORIES = ["electronics","entertainment","fashion","gas_station","grocery","online_marketplace","pharmacy","restaurants","transfer","travel","utilities"]
COUNTRIES = ["AU","BR","CA","DE","ES","FR","GB","IN","IT","US"]
IDENTITIES = ["device_id","ip_address","card_id","merchant_id"]


def raw(event, name):
    value = event.get(name, event.get("raw_payload", {}).get(name))
    if value is None:
        raise ContractError(f"Transaction {event.get('id', '<unknown>')} is missing {name}.")
    return value


def build_features(payload):
    seen = {name:set() for name in IDENTITIES}
    rows = []
    previous_time = None
    for event in payload.get("events", []):
        event_time = timestamp(event["event_time"])
        delta_hours = max(0.0, (event_time-previous_time).total_seconds()/3600) if previous_time else 0.0
        hour = event_time.hour + event_time.minute/60 + event_time.second/3600
        dow = event_time.weekday()
        category = str(raw(event, "merchant_category"))
        country = str(raw(event, "merchant_country"))
        identities = {name:raw(event,name) for name in IDENTITIES}
        row = {
            "has_previous_tx":int(previous_time is not None), "log_delta_hours":math.log1p(delta_hours),
            "hour_sin":math.sin(2*math.pi*hour/24), "hour_cos":math.cos(2*math.pi*hour/24),
            "dow_sin":math.sin(2*math.pi*dow/7), "dow_cos":math.cos(2*math.pi*dow/7),
            "is_new_device":int(identities["device_id"] not in seen["device_id"]),
            "is_new_ip":int(identities["ip_address"] not in seen["ip_address"]),
            "is_new_card":int(identities["card_id"] not in seen["card_id"]),
            "is_new_merchant":int(identities["merchant_id"] not in seen["merchant_id"]),
            "transaction_type_transfer":int(event.get("raw_payload", {}).get("transaction_type", event["type"]) == "transfer"),
            "log_amount":math.log1p(max(0.0,float(event["amount"])))
        }
        row.update({f"merchant_category_{name}":int(category == name) for name in CATEGORIES})
        row.update({f"merchant_country_{name}":int(country == name) for name in COUNTRIES})
        rows.append(row)
        for name,value in identities.items():
            seen[name].add(value)
        previous_time = event_time
    return rows


def reason_codes(payload, snapshot, raw_score):
    events = payload["events"][-10:]
    reasons = []
    if sum(event["type"] == "transfer" for event in events) >= 3:
        reasons.append({"code":"REPEATED_TRANSFERS","description":"Recent sequence contains at least three transfers.","kind":"observation"})
    if len({raw(event,"device_id") for event in events}) > 1:
        reasons.append({"code":"DEVICE_CHANGES","description":"Recent sequence contains multiple devices.","kind":"observation"})
    return reasons
