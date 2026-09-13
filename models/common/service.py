"""Lifecycle and common HTTP runner implementation."""

import ast
from importlib import metadata
import os
from pathlib import Path
import platform
import time
from typing import Any

from fastapi import FastAPI

from models.common.adapter import load_module
from models.common.bundle import check_bundle, json_hash, read_json
from models.common.contracts import ContractError, InsufficientHistory, PredictionRequest, PredictionResponse
from models.common.demo import demo_prediction


class Runner:
    def __init__(self, kind: str, directory: Path, demo: bool = False, trusted: bool = False, require_verified: bool = True) -> None:
        if kind not in {"onboarding", "transaction", "sequence"}:
            raise ValueError("MODEL_KIND must be onboarding, transaction, or sequence.")
        self.kind = kind
        self.directory = directory
        self.demo = demo
        self.trusted = trusted
        self.adapter = None
        self.problem = None
        self.manifest: dict[str, Any] = {}
        self.contract: dict[str, Any] = {}
        try:
            self.manifest = read_json(directory / "manifest.json")
            self.contract = read_json(directory / "feature_contract.json")
            if self.manifest.get("model") != kind:
                raise ContractError("manifest.model differs from MODEL_KIND.")
            if not demo:
                check_bundle(directory, self.manifest, self.contract, require_verified=require_verified)
                if not trusted:
                    raise ContractError("Set ALLOW_TRUSTED_ARTIFACTS=true only for artifacts reviewed and supplied by the team.")
                self.check_environment()
                tree = ast.parse((directory / "pipeline.py").read_text(encoding="utf-8"))
                if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit"} for node in ast.walk(tree)):
                    raise ContractError("pipeline.py contains fitting calls; serving must only transform with frozen state.")
                module = load_module(directory / "adapter.py", f"firstwatch_{kind}_adapter")
                self.adapter = module.Adapter(directory, self.manifest, self.contract)
        except Exception as exc:
            self.problem = f"{type(exc).__name__}: {exc}"
        self.contract_hash = json_hash(self.contract)

    def check_environment(self) -> None:
        environment = self.manifest["environment"]
        if platform.python_version() != environment["python"]:
            raise ContractError(f"Python mismatch: exported {environment['python']}, running {platform.python_version()}.")
        for name, version in environment["libraries"].items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError as exc:
                raise ContractError(f"Missing training dependency {name}=={version}.") from exc
            if actual != version:
                raise ContractError(f"Dependency mismatch: {name}=={actual}; exported with {version}.")

    @property
    def ready(self) -> bool:
        return self.problem is None and (self.demo or self.adapter is not None)

    def health(self) -> dict[str, Any]:
        return {"service": self.kind, "process": "up", "ready": self.ready, "mode": "demo" if self.demo else "external", "status": "demo" if self.demo and self.ready else "ok" if self.ready else "not_ready", "detail": self.problem}

    def metadata(self) -> dict[str, Any]:
        return {
            **self.health(), "model": self.kind,
            "model_version": "synthetic-demo-v1" if self.demo else self.manifest.get("model_version", "pending"),
            "preprocessing_version": "synthetic-observations-v1" if self.demo else self.manifest.get("preprocessing_version", "pending"),
            "contract_hash": self.contract_hash,
            "risk_semantics": "Illustrative deterministic rule score, not a calibrated probability." if self.demo else self.manifest.get("output"),
            "feature_contract": self.contract,
            "required_history": 4 if self.demo and self.kind == "sequence" else self.contract.get("sequence", {}).get("min_history", 0),
            "manifest": self.manifest,
        }

    def payload(self, request: PredictionRequest) -> dict[str, Any]:
        extra_labels = {str(self.manifest.get("target", "")).lower()}
        payload = request.safe_payload(extra_labels)
        if self.demo and payload["context"].get("source") != "synthetic":
            raise ContractError("DEMO_MODE only accepts context.source=synthetic; dataset inference requires an external bundle.")
        return payload

    def validate(self, request: PredictionRequest) -> dict[str, Any]:
        if not self.ready:
            return {"valid": False, "status": "not_ready", "errors": [self.problem or "Runner unavailable."]}
        try:
            payload = self.payload(request)
            if self.demo:
                result = demo_prediction(self.kind, payload)
                snapshot = result["features"]
            else:
                _, snapshot = self.adapter.prepare(payload)
            return {"valid": True, "status": "demo" if self.demo else "ok", "errors": [], "features": snapshot, "contract_hash": self.contract_hash}
        except InsufficientHistory as exc:
            return {"valid": False, "status": "insufficient_history", "errors": [str(exc)]}
        except Exception as exc:
            return {"valid": False, "status": "error", "errors": [str(exc)]}

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        data: dict[str, Any] = {
            "model": self.kind,
            "model_version": "synthetic-demo-v1" if self.demo else self.manifest.get("model_version", "pending"),
            "preprocessing_version": "synthetic-observations-v1" if self.demo else self.manifest.get("preprocessing_version", "pending"),
            "contract_hash": self.contract_hash, "status": "not_ready",
        }
        try:
            if not self.ready:
                data["reason_codes"] = [{"code": "BUNDLE_NOT_READY", "description": self.problem or "Model handoff is pending.", "kind": "observation"}]
            else:
                payload = self.payload(request)
                result = demo_prediction(self.kind, payload) if self.demo else self.adapter.predict(payload)
                data.update(result, status="demo" if self.demo else "ok")
        except InsufficientHistory as exc:
            data.update(status="insufficient_history", reason_codes=[{"code": "INSUFFICIENT_HISTORY", "description": str(exc), "kind": "observation"}])
        except Exception as exc:
            data.update(status="error", risk_score=None, raw_score=None, reason_codes=[{"code": "INFERENCE_ERROR", "description": str(exc), "kind": "observation"}])
        data["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return PredictionResponse.model_validate(data)


def create_app(kind: str | None = None, directory: Path | None = None, demo: bool | None = None) -> FastAPI:
    kind = kind or os.getenv("MODEL_KIND", "onboarding")
    directory = directory or Path(os.getenv("MODEL_DIR", f"models/{kind}"))
    demo = demo if demo is not None else os.getenv("DEMO_MODE", "false").lower() == "true"
    runner = Runner(kind, directory, demo=demo, trusted=os.getenv("ALLOW_TRUSTED_ARTIFACTS", "false").lower() == "true")
    app = FastAPI(title=f"FirstWatch {kind} runner", version="1.0.0")
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runner.health()

    @app.get("/metadata")
    def model_metadata() -> dict[str, Any]:
        return runner.metadata()

    @app.post("/validate")
    def validate(request: PredictionRequest) -> dict[str, Any]:
        return runner.validate(request)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        return runner.predict(request)

    return app
