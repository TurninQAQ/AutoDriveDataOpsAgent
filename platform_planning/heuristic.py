from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from platform_core.config import parse_csv


STAGES = ("precheck", "parser", "segment", "map", "od", "coloration", "occ")
TASK_TYPES = ("release", "reprocess", "test", "debug")


def _unique(items):
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _stage_mentions(text: str) -> list[str]:
    lower = text.lower()
    positions = []
    for stage in STAGES:
        match = re.search(rf"(?<![a-z0-9_]){re.escape(stage)}(?![a-z0-9_])", lower)
        if match:
            positions.append((match.start(), stage))
    return [stage for _, stage in sorted(positions)]


def _parse_pipeline_literal(text: str) -> list[str | list[str]] | None:
    # A compact deterministic syntax for local development:
    # pipeline=precheck,parser,segment,map,[od,occ],coloration
    match = re.search(r"(?:pipeline(?:_stages)?|阶段)\s*[:=：]\s*([^\n;；]+)", text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    # Stop before the next natural-language field. Commas inside the pipeline are
    # valid separators, so only field-name markers terminate this compact syntax.
    raw = re.split(
        r"(?:，|,\s*)(?=(?:数据|dataset|优先级|priority|并发|max_active_runs|gpu|timeout|超时)(?:\s|[:=：]|$))",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    parts = re.findall(r"\[[^\]]+\]|[A-Za-z_]+", raw)
    result: list[str | list[str]] = []
    for part in parts:
        if part.startswith("["):
            group = [s.strip().lower() for s in part[1:-1].split(",") if s.strip()]
            if group:
                result.append(group)
        else:
            result.append(part.strip().lower())
    return result or None


def _extract_paths(text: str) -> list[str]:
    """Extract only paths explicitly described as input data.

    A generic absolute-path scan misclassified output/config destinations as
    datasets. Keep this parser intentionally small and literal; ambiguous paths
    remain available to the model draft or unresolved validation.
    """

    path = r"(/[^\s,，;；。！？?]+)"
    patterns = (
        rf"(?:数据(?:集)?|dataset)\s*(?:在|为|是|路径(?:为)?)?\s*[:=：]?\s*{path}",
        rf"(?:dataset)\s*[:=：]\s*{path}",
        rf"(?:把|用)\s*{path}\s*(?:做|处理|跑|作为)",
        rf"(?:处理|输入|使用)\s*{path}",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(1) for match in re.finditer(pattern, text, re.IGNORECASE))
    return _unique([item.rstrip(".。") for item in found])


def _dataset_names(text: str) -> list[str]:
    names = re.findall(r"(?<![A-Za-z0-9_])(clip[_-][A-Za-z0-9_.-]+)", text, re.IGNORECASE)
    return _unique([item.rstrip(".,，。") for item in names])


def _sanitize_prefix(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "task"
    if not value[0].isalpha():
        value = "task_" + value
    return value


def _mb(value: str, unit: str | None) -> int:
    number = float(value)
    if (unit or "").lower() in {"g", "gb", "gib"}:
        return int(number * 1024)
    return int(number)


class HeuristicTaskDraftParser:
    """Deterministic parser used when no external model is available.

    It intentionally understands a finite, testable task vocabulary. Missing values
    are filled only by the platform defaults file, never by free-form guessing.
    """

    def parse(self, text: str) -> dict[str, Any]:
        lower = text.lower()
        draft: dict[str, Any] = {"explicit_fields": []}

        task_type = next((name for name in TASK_TYPES if re.search(rf"(?<![a-z0-9_]){name}(?![a-z0-9_])", lower)), "")
        if task_type:
            draft["task_type"] = task_type
            draft["explicit_fields"].append("task_type")

        name_match = re.search(
            r"(?:task[_ ]?prefix|task[_ ]?name|任务前缀|任务名)\s*[:=：]?\s*([a-zA-Z][A-Za-z0-9_-]*)",
            text,
            re.IGNORECASE,
        )
        if name_match:
            draft["task_prefix"] = _sanitize_prefix(name_match.group(1))
            draft["explicit_fields"].append("task_prefix")
        elif task_type:
            draft["task_prefix"] = task_type

        priority = re.search(r"(?:priority|优先级)\s*[:=：]?\s*(\d+)", text, re.IGNORECASE)
        if priority:
            draft["priority"] = int(priority.group(1))
            draft["explicit_fields"].append("priority")

        pipeline = _parse_pipeline_literal(text)
        if pipeline:
            draft["pipeline_stages"] = pipeline
            draft["explicit_fields"].append("pipeline_stages")
        elif any(term in lower for term in ("全量", "完整流程", "全流程", "full pipeline", "full-pipeline")):
            draft["pipeline_mode"] = "full"
            draft["explicit_fields"].append("pipeline_stages")
        else:
            only = re.search(r"(?:只运行|只跑|only run)\s*([A-Za-z_、,，和与\s]+)", text, re.IGNORECASE)
            if only:
                mentioned = [s for s in STAGES if re.search(rf"(?<![A-Za-z0-9_]){s}(?![A-Za-z0-9_])", only.group(1), re.IGNORECASE)]
                if mentioned:
                    draft["pipeline_stages"] = mentioned
                    draft["explicit_fields"].append("pipeline_stages")

        concurrent = re.search(
            r"(?:max_active_runs\s*[:=：]?|并发(?:数)?\s*[:=：]?|最多(?:同时)?\s*)(\d+)\s*(?:个\s*)?(?:clip)?",
            text,
            re.IGNORECASE,
        )
        if concurrent:
            draft["max_active_runs"] = int(concurrent.group(1))
            draft["explicit_fields"].append("max_active_runs")

        timeout = re.search(r"(?:timeout|超时(?:时间)?)\s*[:=：]?\s*(\d+)\s*(?:min|分钟)?", text, re.IGNORECASE)
        if timeout:
            draft["timeout_min"] = int(timeout.group(1))
            draft["explicit_fields"].append("datasets.timeout_min")

        gpu_ids = re.search(r"gpu(?:_ids| ids?)\s*[:=：]\s*([0-9,，\s]+)", text, re.IGNORECASE)
        if gpu_ids:
            ids = parse_csv(gpu_ids.group(1))
            draft["gpu_ids"] = ",".join(ids)
            draft["explicit_fields"].append("gpu_ids")

        memory: dict[str, int] = {}
        for stage in STAGES:
            patterns = (
                rf"{stage}\s*(?:显存|memory)?\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(gb|gib|g|mb)?",
                rf"{stage}[^\n,，;；]{{0,8}}?(\d+(?:\.\d+)?)\s*(gb|gib|g|mb)\s*(?:显存|memory)?",
            )
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    memory[stage] = _mb(m.group(1), m.group(2))
                    break
        if memory:
            draft["gpu_stage_memory_mb"] = memory
            draft["explicit_fields"].append("gpu_stage_memory_mb")

        exclusive: list[str] = []
        for match in re.finditer(
            r"([A-Za-z_、,，和与\s]+?)\s*(?:独占(?:GPU)?|exclusive(?:\s+gpu)?)",
            text,
            re.IGNORECASE,
        ):
            exclusive.extend(_stage_mentions(match.group(1)))
        if exclusive:
            draft["exclusive_gpu_stages"] = _unique(exclusive)
            draft["explicit_fields"].append("exclusive_gpu_stages")

        shared: list[str] = []
        for match in re.finditer(
            r"([A-Za-z_、,，和与\s]+?)\s*(?:共享(?:GPU)?|shared(?:\s+gpu)?)",
            text,
            re.IGNORECASE,
        ):
            shared.extend(_stage_mentions(match.group(1)))
        if shared:
            draft["shared_gpu_stages"] = _unique(shared)
            draft["explicit_fields"].append("shared_gpu_stages")

        # Explicit image overrides: image_segment=registry/image:tag
        images: dict[str, str] = {}
        for stage in STAGES:
            m = re.search(
                rf"image_{stage}\s*[:=：]\s*([^\s,，;；]+)", text, re.IGNORECASE
            )
            if m:
                images[stage] = m.group(1).strip()
        if images:
            draft["images"] = images
            draft["explicit_fields"].append("datasets.images")

        paths = _extract_paths(text)
        if paths:
            draft["dataset_paths"] = paths
            draft["explicit_fields"].append("datasets.dataset_path")

        names = _dataset_names(text)
        if names:
            draft["dataset_names"] = names
            draft["explicit_fields"].append("datasets.dataset_name")

        return draft


def derive_dataset_name(path: str, index: int = 0) -> str:
    name = _sanitize_prefix(Path(path).name or f"dataset_{index + 1}")
    return name[:80]
