from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    checkpoint_backend: str
    db_path: Path
    checkpoint_path: Path
    rag_mode: str
    rag_dense_provider: str
    rag_dense_model: str
    rag_embedding_dim: int
    api_host: str
    api_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        state_dir = Path(os.environ.get("AUTODRIVE_STATE_DIR", "./runtime_state")).expanduser()
        return cls(
            state_dir=state_dir,
            checkpoint_backend=os.environ.get("AUTODRIVE_CHECKPOINT_BACKEND", "sqlite").strip().lower(),
            db_path=Path(os.environ.get("AUTODRIVE_DB_PATH", str(state_dir / "autodrive_state.sqlite"))).expanduser(),
            checkpoint_path=Path(os.environ.get("AUTODRIVE_CHECKPOINT_PATH", str(state_dir / "checkpoints.sqlite"))).expanduser(),
            rag_mode=os.environ.get("RAG_MODE", "hybrid").strip().lower(),
            rag_dense_provider=os.environ.get("RAG_DENSE_PROVIDER", "disabled").strip().lower(),
            rag_dense_model=os.environ.get("RAG_DENSE_MODEL", "gemini-embedding-2").strip(),
            rag_embedding_dim=max(128, int(os.environ.get("RAG_EMBEDDING_DIM", "768"))),
            api_host=os.environ.get("API_HOST", "127.0.0.1"),
            api_port=int(os.environ.get("API_PORT", "8080")),
        )

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
