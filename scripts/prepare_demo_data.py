"""Create provenance-preserving candidate fixtures, without fitting or claiming holdout."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re


EVENT_TYPES = {"deposit", "purchase", "transfer", "cash_out", "withdrawal", "payment"}
DEFAULT_LABELS = {
    "fraud_bool", "isFraud", "isFlaggedFraud", "is_fraud", "is_fraudulent",
    "fraud", "fraud_label", "fraud_type", "fraud_pattern", "ground_truth",
    "label", "target",
}


def scalar(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_transaction_mapping(config, headers, require_identity=True):
    columns = config.get("columns", {})
    required = {"account_id", "event_time", "type", "amount"}
    if not required.issubset(columns) or not all(isinstance(columns[key], str) and columns[key] for key in required):
        raise ValueError("Transaction mapping requires account_id, event_time, type and amount columns.")
    mapped = {value for value in columns.values() if isinstance(value, str) and value}
    if not mapped.issubset(headers):
        raise ValueError(f"Column mapping references missing CSV headers: {sorted(mapped - set(headers))}")
    if not config.get("currency") and not columns.get("currency"):
        raise ValueError("Declare a constant currency or a currency column.")
    if not config.get("type_map"):
        raise ValueError("Declare an explicit type_map; transaction categories are never guessed.")
    if config.get("time_format") != "iso8601" and not (
        config.get("time_format") == "step" and config.get("step_seconds", 0) > 0 and config.get("epoch")
    ):
        raise ValueError("Declare ISO8601 timestamps or an epoch and positive step_seconds.")
    if require_identity:
        for key in ("dataset_name", "dataset_version", "source_url"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"Transaction mapping requires nonempty {key} provenance.")
    return columns


def transaction_record(row, row_number, dataset, digest, config, columns):
    instant = row[columns["event_time"]]
    if config["time_format"] == "step":
        epoch = datetime.fromisoformat(config["epoch"].replace("Z", "+00:00"))
        instant = (epoch + timedelta(seconds=float(instant) * config["step_seconds"])).isoformat()
    parsed = datetime.fromisoformat(instant.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Event time at row {row_number} must include a timezone.")
    amount = float(row[columns["amount"]])
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"Invalid amount at row {row_number}.")
    source_type = row[columns["type"]]
    event_type = config["type_map"].get(source_type)
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unmapped transaction type {source_type!r} at row {row_number}.")
    currency = row[columns["currency"]] if columns.get("currency") else config["currency"]
    currency = currency.upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError(f"Invalid ISO currency at row {row_number}.")
    record_id = f"{dataset}:{digest[:12]}:{row_number}"
    if columns.get("event_id"):
        record_id = f"{dataset}:{digest[:12]}:{row[columns['event_id']]}"
    result = {
        "event_id": record_id,
        "account_id": row[columns["account_id"]],
        "event_time": parsed.astimezone(timezone.utc).isoformat(),
        "type": event_type,
        "amount": amount,
        "currency": currency,
        "description": row[columns["description"]] if columns.get("description") else "Dataset replay",
    }
    if columns.get("counterparty_id"):
        result["counterparty_id"] = row[columns["counterparty_id"]] or None
    return result


def prepare(source, output, dataset, limit=30, month=7, mapping=None, account_ids=None, account_limit=None):
    source, output = source.resolve(), output.resolve()
    if output == source or output.exists():
        raise ValueError("Choose a new output path; existing datasets are never overwritten.")
    digest = sha256_file(source)
    records, labels = [], []
    config = json.loads(mapping.read_text(encoding="utf-8")) if mapping else {}
    transaction_dataset = dataset in {"transactions", "momtsim"}
    wanted_accounts = set(account_ids or [])
    selected_accounts = []
    rows_scanned = 0
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if dataset == "baf" and not {"month", "fraud_bool"}.issubset(headers):
            raise ValueError("Expected BAF columns month and fraud_bool.")
        columns = validate_transaction_mapping(config, headers, require_identity=dataset == "transactions") if transaction_dataset else {}
        configured_labels = set(config.get("label_columns", []))
        default_labels_lower = {name.lower() for name in DEFAULT_LABELS}
        configured_labels_lower = {name.lower() for name in configured_labels}
        label_columns = {header for header in headers if header.lower() in default_labels_lower | configured_labels_lower}
        for row_number, row in enumerate(reader):
            rows_scanned += 1
            if dataset == "baf" and int(row["month"]) != month:
                continue
            if transaction_dataset:
                account_id = row[columns["account_id"]]
                if wanted_accounts and account_id not in wanted_accounts:
                    continue
                if account_limit:
                    if account_id not in selected_accounts and len(selected_accounts) < account_limit:
                        selected_accounts.append(account_id)
                    if account_id not in selected_accounts:
                        continue
            raw = {key: scalar(value) for key, value in row.items() if key not in label_columns}
            record_id = f"{dataset}:{digest[:12]}:{row_number}"
            if dataset == "baf":
                records.append({"id": record_id, "source_row": row_number, "raw_payload": raw})
            else:
                record = transaction_record(row, row_number, dataset, digest, config, columns)
                record["raw_payload"] = raw
                record_id = record["event_id"]
                records.append(record)
                if record["account_id"] not in selected_accounts:
                    selected_accounts.append(record["account_id"])
            labels.append({"id": record_id, "labels": {key: scalar(row[key]) for key in label_columns}})
            if not transaction_dataset and len(records) >= limit:
                break
            if transaction_dataset and not wanted_accounts and not account_limit and len(records) >= limit:
                break
    if wanted_accounts and wanted_accounts - set(selected_accounts):
        raise ValueError(f"Requested accounts were not found: {sorted(wanted_accounts - set(selected_accounts))}")
    if not records:
        raise ValueError("No matching rows; output was not created.")
    if transaction_dataset:
        records.sort(key=lambda item: (item["account_id"], item["event_time"], item["event_id"]))
    dataset_name = "Bank Account Fraud Dataset Suite" if dataset == "baf" else config.get("dataset_name", "MoMTSim")
    package = {
        "schema_version": "firstwatch-candidate-v1",
        "provenance": {
            "dataset": dataset_name,
            "dataset_version": config.get("dataset_version", "BAF local LFS copy" if dataset == "baf" else "unrecorded"),
            "source_url": config.get("source_url"),
            "source_file": source.name,
            "sha256": digest,
            "rows_scanned": rows_scanned,
            "row_count": len(records),
            "selected_account_ids": selected_accounts if transaction_dataset else [],
            "complete_account_histories": bool(transaction_dataset and (wanted_accounts or account_limit)),
            "status": "candidate_only",
            "holdout_verified": False,
            "note": "Confirm exclusion against notebook splits. BAF and the transaction dataset have no real shared identity.",
        },
        "records": records,
        "evaluation_only": labels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return package["provenance"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["baf", "transactions", "momtsim"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--limit", type=int, default=30, help="Row limit unless selecting complete account histories.")
    parser.add_argument("--account-id", action="append", dest="account_ids")
    parser.add_argument("--account-limit", type=int, help="Select the first N accounts and scan the full source for their histories.")
    args = parser.parse_args()
    if args.limit < 1 or (args.account_limit is not None and args.account_limit < 1):
        parser.error("limits must be positive")
    if args.dataset == "baf" and (args.account_ids or args.account_limit):
        parser.error("account selection applies only to transaction datasets")
    print(json.dumps(prepare(args.source, args.output, args.dataset, args.limit, args.month, args.mapping, args.account_ids, args.account_limit), indent=2))
