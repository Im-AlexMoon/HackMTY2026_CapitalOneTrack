"""Join unrelated BAF applications and transaction histories using explicit synthetic links."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(applications_path, transactions_path, links_path, output):
    if output.exists():
        raise ValueError("Choose a new output path; demo packages are immutable.")
    applications = load(applications_path)
    transactions = load(transactions_path)
    link_spec = load(links_path)
    app_by_id = {item["id"]: item for item in applications["records"]}
    events_by_account = {}
    for event in transactions["records"]:
        events_by_account.setdefault(event["account_id"], []).append(event)
    result_apps, result_events, audit_links = [], [], []
    demo_ids = set()
    for index, link in enumerate(link_spec.get("links", []), start=1):
        required = {"application_record_id", "transaction_account_id", "demo_account_id", "alias"}
        if not required.issubset(link) or not all(str(link[key]).strip() for key in required):
            raise ValueError(f"Synthetic link {index} is incomplete.")
        if link["demo_account_id"] in demo_ids:
            raise ValueError("demo_account_id values must be unique.")
        demo_ids.add(link["demo_account_id"])
        application = app_by_id.get(link["application_record_id"])
        history = events_by_account.get(link["transaction_account_id"], [])
        if application is None or not history:
            raise ValueError(f"Synthetic link {index} references a missing application or transaction history.")
        history = sorted(history, key=lambda event: (event["event_time"], event["event_id"]))
        first_time = datetime.fromisoformat(history[0]["event_time"].replace("Z", "+00:00"))
        application_time = link.get("application_event_time") or (first_time - timedelta(minutes=1)).isoformat()
        raw_payload = dict(application["raw_payload"])
        raw_payload.setdefault("initial_balance", link.get("initial_balance", 0))
        result_apps.append({
            "application_id": f"demo-application-{index}",
            "account_id": link["demo_account_id"],
            "alias": link["alias"],
            "event_time": application_time,
            "raw_payload": raw_payload,
            "labels": {},
        })
        for event_index, source_event in enumerate(history, start=1):
            event = dict(source_event)
            event["event_id"] = f"{link['demo_account_id']}-event-{event_index}"
            event["account_id"] = link["demo_account_id"]
            result_events.append(event)
        audit_links.append({
            **link,
            "synthetic": True,
            "application_dataset": applications["provenance"]["dataset"],
            "transaction_dataset": transactions["provenance"]["dataset"],
            "transaction_event_count": len(history),
        })
    if not result_apps:
        raise ValueError("At least one explicit synthetic link is required.")
    result_events.sort(key=lambda event: (event["event_time"], event["event_id"]))
    package = {
        "schema_version": "firstwatch-demo-package-v1",
        "status": "candidate_only",
        "holdout_verified": bool(applications["provenance"].get("holdout_verified") and transactions["provenance"].get("holdout_verified")),
        "warning": "BAF and transaction identities are unrelated. Every association below is synthetic and for demonstration only.",
        "sources": {"applications": applications["provenance"], "transactions": transactions["provenance"]},
        "synthetic_links": audit_links,
        "applications": result_apps,
        "events": result_events,
        "evaluation_only": {
            "applications": applications.get("evaluation_only", []),
            "transactions": transactions.get("evaluation_only", []),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"applications": len(result_apps), "events": len(result_events), "holdout_verified": package["holdout_verified"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applications", required=True, type=Path)
    parser.add_argument("--transactions", required=True, type=Path)
    parser.add_argument("--links", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.applications, args.transactions, args.links, args.output), indent=2))
