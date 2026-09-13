"""Replace with a custom adapter only if the notebook needs it."""
from models.common.adapter import ArtifactAdapter

class Adapter(ArtifactAdapter):
    def __init__(self, directory, manifest, contract):
        super().__init__(directory, manifest, contract)
        if list(self.artifact.get("feature_cols", [])) != contract["feature_order"]:
            raise ValueError("Artifact feature_cols differ from the onboarding contract.")
