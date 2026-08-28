from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deploy_ci_cloud_agentv3.platform_backend.rag.sources import KnowledgeSourceConfig, KnowledgeSourceLoader
from deploy_ci_cloud_agentv3.rag.embeddings import EmbeddingProvider


@dataclass
class DenseIndex:
    root: Path

    @property
    def chunks_path(self) -> Path: return self.root / "chunks.jsonl"
    @property
    def embeddings_path(self) -> Path: return self.root / "embeddings.npy"
    @property
    def manifest_path(self) -> Path: return self.root / "manifest.json"

    @staticmethod
    def knowledge_hash(source_dir: Path) -> str:
        return KnowledgeSourceLoader(KnowledgeSourceConfig(source_dir)).fingerprint()

    async def build(self, source_dir: Path, provider: EmbeddingProvider) -> dict:
        loader = KnowledgeSourceLoader(KnowledgeSourceConfig(source_dir))
        chunks = loader.load()
        texts = [f"title: {c.title} | section: {c.section} | text: {c.content}" for c in chunks]
        vectors = await provider.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding count does not match chunk count")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(chunks), provider.dimension):
            raise RuntimeError(f"embedding matrix shape mismatch: {matrix.shape}")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        np.save(self.embeddings_path, matrix, allow_pickle=False)
        manifest = {
            "embedding_model": provider.model_name,
            "embedding_dimension": provider.dimension,
            "chunk_count": len(chunks),
            "knowledge_content_hash": loader.fingerprint(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return manifest

    def manifest(self) -> dict | None:
        if not self.manifest_path.exists(): return None
        try: return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return None

    def is_fresh(self, source_dir: Path, *, model: str, dimension: int) -> bool:
        manifest = self.manifest()
        if not manifest or not self.chunks_path.exists() or not self.embeddings_path.exists(): return False
        if manifest.get("embedding_model") != model or int(manifest.get("embedding_dimension", 0)) != int(dimension): return False
        if manifest.get("knowledge_content_hash") != self.knowledge_hash(source_dir): return False
        try:
            matrix = np.load(self.embeddings_path, allow_pickle=False, mmap_mode="r")
        except Exception:
            return False
        return matrix.ndim == 2 and matrix.shape == (int(manifest.get("chunk_count", -1)), int(dimension))

    def load(self):
        from deploy_ci_cloud_agentv3.platform_backend.rag.models import KnowledgeChunk
        manifest = self.manifest()
        if not manifest: raise RuntimeError("dense index is missing")
        chunks = [KnowledgeChunk.model_validate_json(line) for line in self.chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        matrix = np.load(self.embeddings_path, allow_pickle=False)
        if matrix.shape != (len(chunks), int(manifest["embedding_dimension"])):
            raise RuntimeError("dense index is corrupt or inconsistent")
        return chunks, matrix, manifest
