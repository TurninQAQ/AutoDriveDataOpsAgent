from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .models import KnowledgeChunk


SUPPORTED_SUFFIXES = {".md", ".markdown", ".yaml", ".yml", ".txt"}


@dataclass(frozen=True)
class KnowledgeSourceConfig:
    source_dir: Path
    max_chunk_chars: int = 1800
    overlap_chars: int = 180


class KnowledgeSourceLoader:
    def __init__(self, config: KnowledgeSourceConfig):
        self.config = config

    def source_files(self) -> list[Path]:
        root = self.config.source_dir
        if not root.exists():
            return []
        return sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self.source_files():
            rel = path.relative_to(self.config.source_dir).as_posix()
            data = path.read_bytes()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(data).digest())
        return digest.hexdigest()

    def load(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for path in self.source_files():
            chunks.extend(self._load_file(path))
        return chunks

    def _load_file(self, path: Path) -> list[KnowledgeChunk]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        rel = path.relative_to(self.config.source_dir).as_posix()
        if path.suffix.lower() in {".md", ".markdown"}:
            parts = self._markdown_sections(text)
        else:
            parts = [(path.stem, text)]
        chunks: list[KnowledgeChunk] = []
        for section, body in parts:
            for index, content in enumerate(self._window(body)):
                normalized = content.strip()
                if not normalized:
                    continue
                meaningful = re.sub(r"^#{1,6}\s+.*$", "", normalized, flags=re.MULTILINE).strip()
                if section and len(meaningful) < 20:
                    # Avoid indexing heading-only chunks that can outrank the actual rule body.
                    continue
                content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                stable_key = f"{rel}\n{section}\n{index}\n{content_hash}"
                chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:24]
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_path=rel,
                        title=self._title(text, path),
                        section=section,
                        content=normalized,
                        content_hash=content_hash,
                        metadata={"chunk_index": index, "suffix": path.suffix.lower()},
                    )
                )
        return chunks

    @staticmethod
    def _title(text: str, path: Path) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                if title:
                    return title
        return path.stem.replace("_", " ")

    @staticmethod
    def _markdown_sections(text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current: list[str] = []
        in_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
            heading = None
            if not in_fence:
                match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
                if match:
                    heading = match.group(2).strip()
            if heading is not None:
                if current:
                    sections.append((current_heading, "\n".join(current).strip()))
                current_heading = heading
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append((current_heading, "\n".join(current).strip()))
        return [(heading, body) for heading, body in sections if body]

    def _window(self, text: str) -> list[str]:
        max_chars = max(300, self.config.max_chunk_chars)
        overlap = max(0, min(self.config.overlap_chars, max_chars // 3))
        if len(text) <= max_chars:
            return [text]
        result: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                cut = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(". ", start, end))
                if cut > start + max_chars // 2:
                    end = cut + 1
            result.append(text[start:end])
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return result
