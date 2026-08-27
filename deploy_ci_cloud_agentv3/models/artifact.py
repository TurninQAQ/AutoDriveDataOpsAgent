from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PreparedArtifact(BaseModel):
    artifact_id: str
    sha256: str
    task_prefix: str
    config: dict[str, Any] = Field(default_factory=dict)
    yaml_text: str
