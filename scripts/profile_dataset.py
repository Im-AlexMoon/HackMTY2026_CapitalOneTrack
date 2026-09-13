"""Profile a CSV locally without loading it into memory or exposing row values."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path

try:
    from scripts.prepare_demo_data import sha256_file, validate_transaction_mapping
except ModuleNotFoundError:  # Direct execution from scripts/.
    from prepare_demo_data import sha256_file, validate_transaction_mapping


def value_kind(value):
    if value == "":
        return "null"
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "boolean"
    try:
        number = float(value)
        return "integer" if math.isfinite(number) and number.is_integer() else "number"
    except ValueError:
        return "string"


def profile(source, mapping=None, max_rows=None):
    config = json.loads(mapping.read_text(encoding="utf-8")) if mapping else None
    counters = {}
    accounts = set()
    types, currencies = Counter(), Counter()
    labels = {}
    amount_min = amount_max = None
    time_min = time_max = None
    rows = 0
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        counters = {name: Counter() for name in headers}
        columns = validate_transaction_mapping(config, headers) if config else None
        label_columns = set(config.get("label_columns", [])) if config else set()
        labels = {name: Counter() for name in label_columns}
        for row in reader:
            rows += 1
            for name, value in row.items():
                counters[name][value_kind(value)] += 1
            if columns:
                accounts.add(row[columns["account_id"]])
                types[row[columns["type"]]] += 1
                currency = row[columns["currency"]] if columns.get("currency") else config["currency"]
                currencies[currency] += 1
                amount = float(row[columns["amount"]])
                if math.isfinite(amount):
                    amount_min = amount if amount_min is None else min(amount_min, amount)
                    amount_max = amount if amount_max is None else max(amount_max, amount)
                if config["time_format"] == "iso8601":
                    parsed = datetime.fromisoformat(row[columns["event_time"]].replace("Z", "+00:00"))
                    time_min = parsed if time_min is None else min(time_min, parsed)
                    time_max = parsed if time_max is None else max(time_max, parsed)
                for name in label_columns:
                    labels[name][row[name]] += 1
            if max_rows and rows >= max_rows:
                break
    return {
        "source_file": source.name,
        "sha256": sha256_file(source),
        "rows_profiled": rows,
        "partial": bool(max_rows),
        "columns": [{"name": name, "observed_types": dict(counter), "null_count": counter.get("null", 0)} for name, counter in counters.items()],
        "transaction_summary": None if not config else {
            "dataset_name": config["dataset_name"],
            "dataset_version": config["dataset_version"],
            "distinct_accounts": len(accounts),
            "source_types": dict(types.most_common()),
            "currencies": dict(currencies.most_common()),
            "amount_min": amount_min,
            "amount_max": amount_max,
            "event_time_min": time_min.isoformat() if time_min else None,
            "event_time_max": time_max.isoformat() if time_max else None,
            "evaluation_labels": {name: dict(values.most_common()) for name, values in labels.items()},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    result = profile(args.source, args.mapping, args.max_rows)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
