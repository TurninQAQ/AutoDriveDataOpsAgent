from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import sha256


_LATIN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

DOMAIN_EXPANSIONS = {
    "gpu": ["gpu", "显存", "reservation", "独占", "共享", "nvidia-smi", "资源等待"],
    "显存": ["gpu", "memory", "reservation", "独占", "共享", "oom"],
    "oom": ["gpu", "显存", "out of memory", "cuda", "reservation"],
    "draining": ["draining", "软抢占", "stage boundary", "checkpoint", "recovery"],
    "抢占": ["draining", "软抢占", "priority", "checkpoint", "recovery"],
    "优先级": ["priority", "active", "queued", "draining", "软抢占"],
    "recovery": ["recovery", "checkpoint", "resume", "断点恢复"],
    "恢复": ["recovery", "checkpoint", "resume", "断点恢复"],
    "容器": ["docker", "container", "inspect", "cleanup", "dataset token"],
    "docker": ["docker", "container", "inspect", "cleanup"],
    "airflow": ["airflow", "dagrun", "taskinstance", "scheduler", "metadata database"],
    "失败": ["failed", "validate", "stage", "log", "error"],
    "validate": ["validate", "checkpoint", "result", "stage"],
}


def _cjk_tokens(segment: str) -> list[str]:
    segment = segment.strip()
    if not segment:
        return []
    tokens = [segment]
    # Character n-grams make Chinese retrieval deterministic without jieba.
    for n in (2, 3):
        if len(segment) >= n:
            tokens.extend(segment[i : i + n] for i in range(len(segment) - n + 1))
    return tokens


def tokenize(text: str, expand: bool = False) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    tokens.extend(match.group(0).lower() for match in _LATIN_RE.finditer(lowered))
    for match in _CJK_RE.finditer(lowered):
        tokens.extend(_cjk_tokens(match.group(0)))
    if expand:
        additions: list[str] = []
        for key, values in DOMAIN_EXPANSIONS.items():
            if key in lowered:
                for value in values:
                    additions.extend(tokenize(value, expand=False))
        tokens.extend(additions)
    return [token for token in tokens if token and not token.isspace()]


def hashed_vector(tokens: list[str], dimension: int = 384) -> dict[int, float]:
    counts = Counter(tokens)
    vector: dict[int, float] = {}
    for token, count in counts.items():
        digest = sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        # Sublinear TF reduces dominance from long chunks.
        weight = (1.0 + math.log(float(count))) * sign
        vector[idx] = vector.get(idx, 0.0) + weight
    norm = math.sqrt(sum(v * v for v in vector.values()))
    if norm:
        vector = {idx: value / norm for idx, value in vector.items()}
    return vector


def cosine_sparse(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(idx, 0.0) for idx, value in left.items())
