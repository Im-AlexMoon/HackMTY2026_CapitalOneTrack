"""Adapter for trusted, frozen sklearn bundles and native Keras models.

Every transformation choice is declared in the manifest or exported pipeline.
This module never fits a transformer and never guesses feature definitions.
"""

import importlib.util
import copy
import math
import pickle
from pathlib import Path
from typing import Any

from models.common.bundle import resolve_bundle_path
from models.common.contracts import ContractError, InsufficientHistory, LABEL_KEYS


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"Cannot import {path.name}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_value(value: Any) -> Any:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_value(value.item())
    return value


def validate_row(row: Any, contract: dict[str, Any]) -> list[Any]:
    if not isinstance(row, dict):
        raise ContractError("build_features must return a named feature dictionary or sequence of dictionaries.")
    names = contract["feature_order"]
    if set(row) != set(names):
        raise ContractError(f"Feature keys differ from the contract: missing={sorted(set(names) - set(row))}, extra={sorted(set(row) - set(names))}.")
    result = []
    for field in contract["features"]:
        name = field["name"]
        if name.lower() in LABEL_KEYS:
            raise ContractError(f"Label-like feature {name} is forbidden.")
        value = row[name]
        if value is None:
            if not field["nullable"]:
                raise ContractError(f"{name} cannot be null.")
            result.append(value)
            continue
        dtype = field["dtype"]
        valid = {
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
        }[dtype]
        if not valid or (isinstance(value, (float, int)) and not math.isfinite(value)):
            raise ContractError(f"{name} must be a finite {dtype}; coercion must be explicit in pipeline.py.")
        if "categories" in field and value not in field["categories"] and field["unknown_category"] == "reject":
            raise ContractError(f"Unknown category for {name}: {value!r}.")
        result.append(value)
    return result


def normalize_score(raw: float, output: dict[str, Any]) -> float:
    import numpy as np

    if not math.isfinite(raw):
        raise ContractError("Model returned a nonfinite score.")
    normalization = output["normalization"]
    kind = normalization["kind"]
    if kind == "identity":
        risk = raw
    elif kind == "affine":
        low, high = normalization["low"], normalization["high"]
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ContractError("Affine normalization requires finite low < high.")
        risk = (raw - low) / (high - low)
        if normalization.get("clip", False):
            risk = max(0.0, min(1.0, risk))
    elif kind == "logistic":
        margin = normalization["slope"] * raw + normalization["intercept"]
        risk = 1 / (1 + math.exp(-max(-700, min(700, margin))))
    elif kind == "ecdf":
        reference = np.asarray(normalization["reference"], dtype=float)
        if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all() or np.any(reference[1:] < reference[:-1]):
            raise ContractError("ECDF reference must be sorted, finite, and validation-only.")
        risk = float(np.searchsorted(reference, raw, side="right") / len(reference))
    else:
        raise ContractError("Unknown normalization kind.")
    if output.get("higher_is_risk", True) is False:
        risk = 1 - risk
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ContractError("Recorded score mapping produced risk outside 0–1.")
    return float(risk)


class ArtifactAdapter:
    def __init__(self, directory: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> None:
        self.directory = directory
        self.manifest = manifest
        self.contract = contract
        self.pipeline = load_module(directory / "pipeline.py", f"firstwatch_{manifest['model']}_pipeline")
        if not callable(getattr(self.pipeline, "build_features", None)):
            raise ContractError("pipeline.py must export build_features(payload).")
        artifact = next((item for item in manifest["artifacts"] if item.get("role") == "model"), manifest["artifacts"][0])
        path = resolve_bundle_path(directory, artifact["path"])
        artifact_format = manifest["artifact_format"]
        if artifact_format == "pickle":
            with path.open("rb") as stream:
                self.artifact = pickle.load(stream)
        elif artifact_format == "joblib":
            import joblib

            self.artifact = joblib.load(path)
        elif artifact_format == "keras":
            import keras

            self.artifact = keras.models.load_model(path, compile=False, safe_mode=True)
        elif artifact_format == "custom":
            self.artifact = self.pipeline.load_artifact(directory, manifest)
        else:
            raise ContractError("Use pickle, joblib, keras, or custom artifact_format.")
        self.model = self.artifact.get("model") if isinstance(self.artifact, dict) else self.artifact
        if self.model is None:
            raise ContractError("Exported dictionary must contain model.")
        if isinstance(self.artifact, dict) and "feature_names" in self.artifact:
            if list(self.artifact["feature_names"]) != contract["feature_order"]:
                raise ContractError("Artifact feature_names differ from feature_contract.feature_order.")
        self.preprocessor = None
        if isinstance(self.artifact, dict):
            # Training bundles use either name. Supporting both keeps the HTTP
            # boundary stable without rewriting a trusted serialized artifact.
            self.preprocessor = self.artifact.get("preprocessor", self.artifact.get("encoder"))
        if manifest["artifact_format"] == "keras" and manifest["preprocessing_mode"] == "bundle_transformer":
            scaler = next((item for item in manifest["artifacts"] if item.get("role") == "preprocessor"), None)
            if scaler is None:
                raise ContractError("Native Keras bundle_transformer requires an exported preprocessor artifact.")
            with resolve_bundle_path(directory, scaler["path"]).open("rb") as stream:
                self.preprocessor = pickle.load(stream)
        self.output = copy.deepcopy(manifest["output"])
        normalization = self.output.get("normalization", {})
        reference_role = normalization.get("reference_artifact")
        if normalization.get("kind") == "ecdf" and reference_role:
            reference = next((item for item in manifest["artifacts"] if item.get("role") == reference_role), None)
            if reference is None:
                raise ContractError(f"Missing ECDF reference artifact role: {reference_role}.")
            reference_path = resolve_bundle_path(directory, reference["path"])
            if reference.get("format") != "joblib":
                raise ContractError("ECDF reference artifact must use joblib format.")
            import joblib

            reference_bundle = joblib.load(reference_path)
            reference_key = normalization.get("reference_key", "reference_errors")
            if not isinstance(reference_bundle, dict) or reference_key not in reference_bundle:
                raise ContractError(f"ECDF reference artifact is missing {reference_key}.")
            self.output["normalization"]["reference"] = reference_bundle[reference_key]
            expected_features = reference_bundle.get("features")
            if expected_features is not None and list(expected_features) != contract["feature_order"]:
                raise ContractError("Sequence artifact features differ from feature_contract.feature_order.")
            expected_steps = reference_bundle.get("timesteps")
            if expected_steps is not None and expected_steps != contract.get("sequence", {}).get("length"):
                raise ContractError("Sequence artifact timesteps differ from the feature contract.")

    def prepare(self, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        import numpy as np
        import pandas as pd

        sequence = self.contract.get("sequence", {})
        if self.contract["input_layout"] == "sequence" and len(payload["events"]) < sequence["min_history"]:
            raise InsufficientHistory(f"Requires at least {sequence['min_history']} observed events.")
        derived = self.pipeline.build_features(payload)
        if self.contract["input_layout"] == "sequence":
            if not isinstance(derived, list) or len(derived) < sequence["min_history"]:
                raise InsufficientHistory("build_features did not produce the required sequence history.")
            rows = [validate_row(row, self.contract) for row in derived[-sequence["length"]:]]
            if len(rows) < sequence["length"] and sequence["padding"] == "none":
                raise InsufficientHistory(f"Requires a complete window of {sequence['length']} rows; padding is disabled.")
        else:
            rows = [validate_row(derived, self.contract)]
        frame = pd.DataFrame(rows, columns=self.contract["feature_order"])
        model_input = frame if self.manifest["input_format"] == "dataframe" else frame.to_numpy()
        mode = self.manifest["preprocessing_mode"]
        if mode == "embedded_pipeline":
            steps = getattr(self.model, "steps", None)
            if not steps:
                raise ContractError("embedded_pipeline requires a fitted sklearn-compatible Pipeline.")
            transformed = self.model[:-1].transform(model_input) if len(steps) > 1 else model_input
        elif mode == "bundle_transformer":
            if self.preprocessor is None or not hasattr(self.preprocessor, "transform"):
                raise ContractError("The fitted preprocessor is missing from the bundle.")
            transformed = self.preprocessor.transform(model_input)
        elif mode == "pipeline_module":
            transformed = self.pipeline.transform(model_input, self.artifact)
        elif mode == "none":
            transformed = model_input
        else:
            raise ContractError("preprocessing_mode must be explicit.")
        if mode != "embedded_pipeline" and isinstance(self.artifact, dict) and self.artifact.get("selector") is not None:
            transformed = self.artifact["selector"].transform(transformed)
        if self.contract["input_layout"] == "sequence":
            transformed = np.asarray(transformed, dtype=float)
            if transformed.ndim != 2:
                raise ContractError("Sequence preprocessing must return [time, features].")
            missing = sequence["length"] - len(transformed)
            if missing > 0:
                padding = np.full((missing, transformed.shape[1]), sequence["padding_value"])
                transformed = np.concatenate([padding, transformed] if sequence["padding"] == "pre" else [transformed, padding])
            transformed = transformed[np.newaxis, :, :]
        snapshot = {
            "feature_order": self.contract["feature_order"], "raw_vector": json_value(rows),
            "model_vector": json_value(transformed.values if hasattr(transformed, "values") else transformed),
            "model_feature_order": self.contract.get("model_feature_order"),
            "input_layout": self.contract["input_layout"],
        }
        return transformed, snapshot

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        values, snapshot = self.prepare(payload)
        estimator = self.model.steps[-1][1] if self.manifest["preprocessing_mode"] == "embedded_pipeline" else self.model
        output = self.output
        method = output["method"]
        if method == "predict_proba":
            classes = list(getattr(estimator, "classes_", []))
            if output["positive_class"] not in classes:
                raise ContractError("Declared positive_class is absent from estimator.classes_.")
            raw = float(estimator.predict_proba(values)[0][classes.index(output["positive_class"])])
        elif method == "reconstruction_mse":
            if self.manifest["artifact_format"] != "keras" or self.contract["input_layout"] != "sequence":
                raise ContractError("reconstruction_mse requires a native Keras sequence bundle.")
            if self.contract["sequence"]["padding"] != "none":
                raise ContractError("Masked or padded reconstruction requires a custom score function matching the notebook.")
            reconstructed = np.asarray(estimator.predict(values, verbose=0))
            if reconstructed.shape != values.shape:
                raise ContractError("Reconstruction shape differs from model input.")
            raw = float(np.mean(np.square(values - reconstructed)))
        elif method == "custom":
            raw = float(self.pipeline.score(estimator, values, payload))
        else:
            raw_values = np.asarray(getattr(estimator, method)(values))
            if raw_values.size != 1:
                raise ContractError("Expected exactly one score; provide pipeline.score for custom output shapes.")
            raw = float(raw_values.reshape(-1)[0])
        reasons = self.pipeline.reason_codes(payload, snapshot, raw) if callable(getattr(self.pipeline, "reason_codes", None)) else []
        return {"raw_score": raw, "risk_score": normalize_score(raw, output), "features": snapshot, "reason_codes": reasons}
