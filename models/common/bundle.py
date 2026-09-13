"""Static bundle checks and content addressing, without deserializing artifacts."""

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from models.common.contracts import ContractError


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path.name} must contain a JSON object.")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_bundle_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ContractError("Artifact paths must remain inside MODEL_DIR.")
    return path


def verify_artifacts(directory: Path, manifest: dict[str, Any]) -> list[Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError("manifest.artifacts requires the exact file paths and SHA-256 hashes.")
    paths = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
            raise ContractError("Each artifact requires path and sha256.")
        path = resolve_bundle_path(directory, artifact["path"])
        if not path.is_file():
            raise ContractError(f"Missing artifact: {artifact['path']}")
        if sha256_file(path) != artifact["sha256"]:
            raise ContractError(f"SHA-256 mismatch: {artifact['path']}")
        paths.append(path)
    return paths


def bundle_hash(directory: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> str:
    files = ["adapter.py", "pipeline.py", "requirements.lock", "tests/golden_cases.json"]
    hashes = {name: sha256_file(directory / name) for name in files}
    hashes["artifacts"] = {str(path.relative_to(directory)): sha256_file(path) for path in verify_artifacts(directory, manifest)}
    return json_hash({"manifest": manifest, "contract": contract, "files": hashes})


def check_bundle(directory: Path, manifest: dict[str, Any], contract: dict[str, Any], require_verified: bool = True) -> None:
    if manifest.get("status") not in {"candidate", "ready"}:
        raise ContractError("Model handoff is pending; complete manifest and preprocessing before enabling this runner.")
    for key in ("model", "model_version", "preprocessing_version", "dataset", "target", "artifact_format", "environment", "output"):
        if not manifest.get(key):
            raise ContractError(f"Missing manifest.{key}.")
    provenance = manifest.get("dataset_provenance", {})
    if not provenance.get("version") or not provenance.get("source_url") or not provenance.get("split_reference"):
        raise ContractError("dataset_provenance requires version, source_url and split_reference.")
    source_files = provenance.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise ContractError("dataset_provenance.files requires at least one source file and SHA-256.")
    for item in source_files:
        if not item.get("name"):
            raise ContractError("Each dataset source file requires a name.")
        digest = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest) and not item.get("not_available_reason"):
            raise ContractError("Each dataset source file requires a lowercase SHA-256 or an explicit not_available_reason.")
    environment = manifest["environment"]
    if manifest.get('input_format') not in {'dataframe', 'numpy'}:
        raise ContractError('input_format must explicitly be dataframe or numpy.')
    if manifest.get('preprocessing_mode') not in {'embedded_pipeline', 'bundle_transformer', 'pipeline_module', 'none'}:
        raise ContractError('preprocessing_mode must declare exactly how frozen preprocessing runs.')
    if not environment.get("python") or not environment.get("libraries"):
        raise ContractError("Record training Python and library versions in manifest.environment.")
    if contract.get("status") != "ready":
        raise ContractError("feature_contract.status must be ready.")
    features = contract.get("features", [])
    names = [field.get("name") for field in features]
    if not names or len(names) != len(set(names)) or contract.get("feature_order") != names:
        raise ContractError("features must be nonempty, unique, and in the exact feature_order.")
    for field in features:
        if field.get("dtype") not in {"number", "integer", "string", "boolean"} or not isinstance(field.get("nullable"), bool):
            raise ContractError("Each feature requires dtype and an explicit nullable boolean.")
        if "categories" in field and field.get("unknown_category") not in {"reject", "allow"}:
            raise ContractError("Categorical features require unknown_category=reject or allow.")
    if any(name.lower() == str(manifest["target"]).lower() for name in names):
        raise ContractError("The target cannot be included in feature_order.")
    if contract.get("input_layout") not in {"tabular", "sequence"}:
        raise ContractError("input_layout must be tabular or sequence.")
    if contract["input_layout"] == "sequence":
        sequence = contract.get("sequence", {})
        if not isinstance(sequence.get("length"), int) or sequence["length"] < 1:
            raise ContractError("Sequence contract requires a positive length.")
        if sequence.get("padding") not in {"none", "pre", "post"} or not isinstance(sequence.get("min_history"), int):
            raise ContractError("Sequence contract requires explicit padding and min_history.")
        if not 1 <= sequence["min_history"] <= sequence["length"]:
            raise ContractError("sequence.min_history must lie between 1 and length.")
        if sequence["padding"] != "none" and "padding_value" not in sequence:
            raise ContractError("Padded sequences require the exported padding_value.")
    output = manifest["output"]
    if output.get("method") not in {"predict_proba", "predict", "decision_function", "reconstruction_mse", "custom"}:
        raise ContractError("output.method is not supported.")
    normalization = output.get("normalization", {})
    if normalization.get("kind") not in {"identity", "affine", "logistic", "ecdf"}:
        raise ContractError("An explicit output normalization is required.")
    if normalization["kind"] == "ecdf" and not normalization.get("reference") and not normalization.get("reference_artifact"):
        raise ContractError("ECDF requires a validation-only reference distribution or reference artifact.")
    if output["method"] == "predict_proba" and "positive_class" not in output:
        raise ContractError("predict_proba requires the recorded positive_class.")
    verify_artifacts(directory, manifest)
    if require_verified:
        verification_path = directory / "verification.json"
        if not verification_path.is_file():
            raise ContractError("Run validate_model_bundle.py --write-verification after golden parity tests.")
        verification = read_json(verification_path)
        if verification.get("status") != "passed" or verification.get("cases", 0) < 2 or verification.get("bundle_hash") != bundle_hash(directory, manifest, contract):
            raise ContractError("Golden verification is missing, stale, or has fewer than two cases.")
