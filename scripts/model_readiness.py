"""Report model handoff readiness without importing or deserializing model artifacts."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common.bundle import check_bundle, read_json, verify_artifacts


MODEL_NAMES = ("onboarding", "transaction", "sequence")


def inspect_model(directory):
    manifest = read_json(directory / "manifest.json")
    contract = read_json(directory / "feature_contract.json")
    cases = read_json(directory / "tests" / "golden_cases.json").get("cases", [])
    pipeline_text = (directory / "pipeline.py").read_text(encoding="utf-8").lower()
    provenance = manifest.get("dataset_provenance", {})
    provenance_text = json.dumps(provenance).lower()
    checks = {
        "dataset provenance confirmed": bool(
            manifest.get("dataset") and provenance.get("version") and provenance.get("source_url")
            and provenance.get("files") and provenance.get("split_reference") and "pending" not in provenance_text
        ),
        "target declared": bool(manifest.get("target")),
        "environment frozen": bool(manifest.get("environment", {}).get("python") and manifest.get("environment", {}).get("libraries")),
        "artifact hashes valid": False,
        "feature contract ready": contract.get("status") == "ready" and bool(contract.get("feature_order")),
        "serving pipeline implemented": "handoff pending" not in pipeline_text and "not implemented" not in pipeline_text,
        "at least two golden integration cases": len(cases) >= 2,
        "content-hashed verification current": False,
    }
    try:
        verify_artifacts(directory, manifest)
        checks["artifact hashes valid"] = True
    except Exception:
        pass
    validation_error = None
    try:
        check_bundle(directory, manifest, contract, require_verified=True)
        checks["content-hashed verification current"] = True
    except Exception as exc:
        validation_error = str(exc)
    return {
        "model": manifest.get("model", directory.name),
        "bundle_status": manifest.get("status", "unknown"),
        "ready": all(checks.values()),
        "checks": checks,
        "validation_error": validation_error,
    }


def report(root):
    models = [inspect_model(root / "models" / name) for name in MODEL_NAMES]
    return {"all_ready": all(model["ready"] for model in models), "models": models}


def markdown(value):
    lines = ["# Model readiness report", ""]
    for model in value["models"]:
        lines.extend([f"## {model['model']}", ""])
        for label, passed in model["checks"].items():
            lines.append(f"- [{'x' if passed else ' '}] {label}")
        if model["validation_error"]:
            lines.extend(["", f"Blocking validation: `{model['validation_error']}`"])
        lines.append("")
    lines.append(f"Overall: **{'ready' if value['all_ready'] else 'waiting for model handoff'}**")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    rendered = markdown(result) if args.format == "markdown" else json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.require_ready and not result["all_ready"]:
        raise SystemExit(2)
