"""Create deterministic integration baselines from the reviewed model bundles.

These receipts exercise the canonical request, ported preprocessing and frozen
artifact together. They do not replace an independent Colab parity export.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common.contracts import PredictionRequest
from models.common.service import Runner


def load_scenarios(path):
    spec = importlib.util.spec_from_file_location("firstwatch_scenarios", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_for(kind, application, events):
    as_of = events[-1]["event_time"] if events else application["event_time"]
    return PredictionRequest(
        entity_id=events[-1]["account_id"] if events else application["account_id"],
        as_of=as_of,
        application=application["raw_payload"],
        events=events,
        context={"source": "synthetic", "portfolio_events": events},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("onboarding", "transaction", "sequence"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("scenarios", type=Path)
    args = parser.parse_args()
    scenarios = load_scenarios(args.scenarios)
    runner = Runner(args.kind, args.directory, demo=False, trusted=True, require_verified=False)
    if not runner.ready:
        raise RuntimeError(runner.problem)

    if args.kind == "onboarding":
        source = [
            ("ordinary_application", scenarios.application("golden_safe", "Golden Safe", 1, 22)),
            ("high_risk_application", scenarios.application("golden_risky", "Golden Risky", 2, 95, 8, True)),
        ]
        requests = [(name, request_for(args.kind, item, [])) for name, item in source]
    else:
        applications = {item["account_id"]: item for item in scenarios.initial_accounts()}
        legitimate = scenarios.events_for("legitimate")
        sleeper = [item for item in scenarios.events_for("sleeper_bustout") if item.get("kind") == "event"]
        requests = [
            ("ordinary_history", request_for(args.kind, applications["acct_ava"], legitimate)),
            ("sleeper_cashout", request_for(args.kind, applications["acct_sleeper"], sleeper)),
        ]

    cases = []
    for name, request in requests:
        result = runner.predict(request).model_dump(mode="json")
        if result["status"] != "ok":
            raise RuntimeError(f"{name} did not score: {result}")
        cases.append({
            "name": name, "input": request.model_dump(mode="json"), "atol": 1e-6,
            "expected": {"risk_score": result["risk_score"], "model_vector": result["features"]["model_vector"]},
        })
    document = {
        "status": "integration_baseline",
        "origin": "Reviewed delivered artifact plus ported notebook preprocessing; independent Colab parity remains recommended.",
        "cases": cases,
    }
    destination = args.directory / "tests" / "golden_cases.json"
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.kind, "cases": len(cases), "path": str(destination)}))


if __name__ == "__main__":
    main()
