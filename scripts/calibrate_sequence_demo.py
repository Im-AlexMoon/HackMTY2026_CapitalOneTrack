"""Find deterministic in-distribution sequence parameters for the demo story."""
from datetime import datetime, timedelta, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from models.common.service import Runner


def payload(delta_hours, amount, category, country, hour, rotate_merchant):
    start = datetime(2026, 9, 1, hour, tzinfo=timezone.utc)
    events = []
    for index in range(12):
        merchant = f"merchant-{index}" if rotate_merchant else "merchant-main"
        events.append({
            "id": f"cal-{index}", "event_time": (start + timedelta(hours=index * delta_hours)).isoformat(),
            "type": "purchase", "amount": amount, "raw_payload": {
                "transaction_type": "purchase", "card_id": "card-main", "device_id": "device-main",
                "ip_address": "198.51.100.22", "merchant_id": merchant,
                "merchant_category": category, "merchant_country": country,
            },
        })
    return {"entity_id": "calibration", "as_of": events[-1]["event_time"], "application": {},
            "events": events, "context": {"source": "synthetic"}}


def main():
    runner = Runner("sequence", Path("models/sequence"), demo=False, trusted=True, require_verified=False)
    if not runner.ready:
        raise RuntimeError(runner.problem)
    candidates = []
    arrays = []
    for values in product(
        (1, 6, 24, 72, 168, 336, 720), (1, 5, 10, 25, 50, 100, 500),
        ("grocery", "restaurants", "gas_station", "online_marketplace"),
        ("US", "CA", "GB", "BR"), (0, 6, 12, 18), (False, True),
    ):
        item = payload(*values)
        values_array, _ = runner.adapter.prepare(item)
        candidates.append(item)
        arrays.append(values_array[0])
    values = np.asarray(arrays)
    reconstructed = np.asarray(runner.adapter.model.predict(values, batch_size=256, verbose=0))
    errors = np.mean(np.square(values - reconstructed), axis=(1, 2))
    for index in np.argsort(errors)[:10]:
        item = candidates[int(index)]
        first, second = item["events"][:2]
        print(json.dumps({
            "mse": float(errors[index]), "risk": runner.adapter.predict(item)["risk_score"],
            "delta_hours": (datetime.fromisoformat(second["event_time"]) - datetime.fromisoformat(first["event_time"])).total_seconds() / 3600,
            "amount": first["amount"], "category": first["raw_payload"]["merchant_category"],
            "country": first["raw_payload"]["merchant_country"],
            "hour": datetime.fromisoformat(first["event_time"]).hour,
            "rotate_merchant": first["raw_payload"]["merchant_id"] != second["raw_payload"]["merchant_id"],
        }))


if __name__ == "__main__":
    main()
