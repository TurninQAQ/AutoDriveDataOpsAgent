from __future__ import annotations

from deploy_ci_cloud_agentv3.models.artifact import PreparedArtifact


class ArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, PreparedArtifact] = {}

    def put(self, artifact: PreparedArtifact) -> PreparedArtifact:
        self._items[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> PreparedArtifact:
        try:
            return self._items[artifact_id]
        except KeyError as exc:
            raise KeyError(f"prepared artifact not found: {artifact_id}") from exc
