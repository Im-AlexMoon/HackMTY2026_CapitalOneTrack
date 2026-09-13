"""Stage, build and content-verify every trusted model bundle locally."""
from pathlib import Path
import subprocess
import sys

from stage_model_artifacts import main as stage_artifacts


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("onboarding", "transaction", "sequence")


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    stage_artifacts()
    run(["docker", "compose", "build", *MODELS])
    for model in MODELS:
        directory = (ROOT / "models" / model).resolve()
        # docker-compose.yml fixes the project name to ``firstwatch``, so these
        # tags are stable even before any service container has been created.
        image = f"firstwatch-{model}"
        run([
            "docker", "run", "--rm", "--network", "none", "--user", "root",
            "-v", f"{directory}:/app/models/{model}:rw", image,
            "python", "/app/scripts/validate_model_bundle.py", f"/app/models/{model}",
            "--trusted", "--write-verification",
        ])
    print("All three model bundles are staged and content-verified.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
