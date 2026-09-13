"""Generate a review-required transaction mapping draft from actual CSV headers."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

try:
    from scripts.prepare_demo_data import sha256_file
except ModuleNotFoundError:  # Direct execution from scripts/.
    from prepare_demo_data import sha256_file


HINTS = {
    "event_id": ("transaction_id", "event_id", "id"),
    "account_id": ("account_id", "customer_id", "user_id", "client_id"),
    "event_time": ("transaction_timestamp", "timestamp", "event_time", "transaction_time", "datetime"),
    "type": ("transaction_type", "type", "operation_type", "category"),
    "amount": ("transaction_amount", "amount", "value"),
    "currency": ("currency", "currency_code"),
    "counterparty_id": ("merchant_id", "counterparty_id", "beneficiary_id", "recipient_id"),
    "description": ("description", "merchant_category", "memo"),
}


def detect(headers, candidates):
    lookup = {header.lower(): header for header in headers}
    return next((lookup[name] for name in candidates if name in lookup), None)


def create_draft(source, dataset_name, dataset_version, source_url, sample_rows=100000):
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        columns = {name: detect(headers, choices) for name, choices in HINTS.items()}
        columns = {name: value for name, value in columns.items() if value}
        observed_types = Counter()
        type_column = columns.get("type")
        for index, row in enumerate(reader):
            if type_column:
                observed_types[row[type_column]] += 1
            if index + 1 >= sample_rows:
                break
    labels = [header for header in headers if any(token in header.lower() for token in ("fraud", "label", "target", "ground_truth"))]
    return {
        "status": "draft_requires_review",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "source_url": source_url,
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "columns": columns,
        "currency": None if "currency" in columns else "USD",
        "time_format": "iso8601",
        "type_map": {name: None for name in observed_types},
        "observed_transaction_types": dict(observed_types.most_common()),
        "label_columns": labels,
        "review": [
            "Confirm every detected column against the training notebook.",
            "Map every observed type to a FirstWatch event type; null values are invalid.",
            "Confirm timestamp timezone/format and currency semantics.",
            "Confirm all target and post-outcome columns are listed in label_columns.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rows", type=int, default=100000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    draft = create_draft(args.source, args.dataset_name, args.dataset_version, args.source_url, args.sample_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
    print(f"Draft created at {args.output}; review is mandatory before import.")
