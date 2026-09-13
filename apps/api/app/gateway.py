"""HTTP model boundary; the orchestrator never deserializes artifacts."""

import asyncio
from datetime import datetime, timezone
import time

import httpx

from .contracts import Prediction


LABEL_KEYS = {
    "labels", "label", "target", "ground_truth", "fraud_bool", "isfraud",
    "is_fraud", "is_fraudulent", "fraud", "fraud_label", "fraud_type",
    "fraud_pattern", "isflaggedfraud",
}


def inference_payload(value):
    """Preserve raw data in storage but deny evaluation labels to every runner."""
    if isinstance(value, dict):
        return {key: inference_payload(item) for key, item in value.items() if key.lower() not in LABEL_KEYS}
    if isinstance(value, list):
        return [inference_payload(item) for item in value]
    return value


class ModelGateway:
    def __init__(self, urls, transport=None, demo_mode=False):
        self.urls = urls
        self.demo_mode = demo_mode
        self.client = httpx.AsyncClient(timeout=8.0, transport=transport)
        self.statuses = {name: {"model": name, "status": "not_ready", "model_version": "pending"} for name in urls}

    async def metadata(self, attempts=1, delay=0.0):
        async def get(name, url):
            try:
                response = await self.client.get(f"{url}/metadata")
                response.raise_for_status()
                value = response.json()
                self.statuses[name] = {**value, "model": name}
                return True
            except (httpx.HTTPError, ValueError):
                self.statuses[name] = {
                    "model": name, "status": "not_ready", "model_version": "unavailable",
                    "detail": "Runner metadata is unavailable.",
                }
                return False
        for attempt in range(attempts):
            results = await asyncio.gather(*(get(name, url) for name, url in self.urls.items()))
            if all(results) or attempt == attempts - 1:
                break
            await asyncio.sleep(delay)

    def readiness(self):
        """Return a small, stable view of real-bundle readiness for API and UI consumers."""
        models = []
        for name in self.urls:
            item = self.statuses.get(name, {})
            manifest = item.get("manifest") or {}
            runtime_status = item.get("status", "not_ready")
            bundle_status = manifest.get("status", "unknown")
            models.append({
                "model": name,
                "service_status": runtime_status,
                "bundle_status": bundle_status,
                # Runner readiness is independent from whether the orchestrator
                # is replaying a curated demo or accepting external ingestion.
                "ready_for_external": item.get("mode") == "external" and runtime_status == "ok",
                "model_version": item.get("model_version", "unavailable"),
                "preprocessing_version": item.get("preprocessing_version"),
                "contract_hash": item.get("contract_hash"),
                "required_history": item.get("required_history", 0),
                "detail": item.get("detail"),
            })
        return {
            "mode": "demo" if self.demo_mode else "external",
            "all_ready": all(model["ready_for_external"] for model in models),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "models": models,
        }

    async def predict(self, name, payload):
        start = time.perf_counter()
        try:
            response = await self.client.post(f"{self.urls[name]}/predict", json=inference_payload(payload))
            response.raise_for_status()
            prediction = Prediction.model_validate(response.json())
            if prediction.model != name:
                raise ValueError("Runner returned the wrong model identity")
            if prediction.status not in {"ok", "demo"}:
                prediction.risk_score = None
            if prediction.status == "demo" and not self.demo_mode:
                raise ValueError("Demo runner cannot score external-mode requests")
        except (httpx.HTTPError, ValueError, KeyError) as error:
            prediction = Prediction(model=name, status="error", reason_codes=[{
                "code": "RUNNER_UNAVAILABLE", "description": f"{name.title()} signal unavailable ({type(error).__name__}).", "kind": "observation"
            }])
        result = prediction.model_dump()
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return result

    async def close(self):
        await self.client.aclose()
