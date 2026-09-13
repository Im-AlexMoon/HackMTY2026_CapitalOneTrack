"""Copy reviewed training outputs into isolated runner bundles and verify hashes."""

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "onboarding": {"artifact/model.joblib": ROOT / "EDA/data/Models_BAF/model_bundle_temporal_sep.joblib"},
    "transaction": {"artifact/model.joblib": ROOT / "EDA/data/Model_transactions/fraud_detection_pipeline_v1.joblib"},
    "sequence": {
        "artifact/model.keras": ROOT / "EDA/data/Model_lstm_ae/lstm_model.keras",
        "artifact/reference.joblib": ROOT / "EDA/data/Model_lstm_ae/lstm_autoencoder_artifacts.joblib",
    },
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    for model, files in SOURCES.items():
        directory = ROOT / "models" / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
        for relative, source in files.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            actual = digest(source)
            if actual != expected[relative]:
                raise ValueError(f"Unexpected SHA-256 for {source}: {actual}")
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(f"{model}: {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)} [{actual[:12]}]")


if __name__ == "__main__":
    main()
