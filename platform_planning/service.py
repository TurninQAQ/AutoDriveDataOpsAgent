from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from platform_core.config import (
    MAX_TASK_PREFIX_LENGTH,
    flatten_stage_groups,
    normalize_pipeline_stages,
    normalize_task_priority_config,
    validate_config,
    validate_task_name,
)
from platform_core.errors import TaskConfigError

from .defaults import TaskPlanningDefaults
from .heuristic import HeuristicTaskDraftParser, derive_dataset_name
from .models import DatasetSpec, TaskPlanningResult, TaskSpec, ValidationIssue


_DETERMINISTIC_DRAFT_FIELDS = (
    "task_type",
    "task_prefix",
    "priority",
    "dataset_paths",
    "dataset_names",
    "pipeline_stages",
    "pipeline_mode",
    "timeout_min",
    "max_active_runs",
    "gpu_ids",
    "gpu_stage_memory_mb",
    "exclusive_gpu_stages",
    "shared_gpu_stages",
    "images",
)

_FIELD_EXPLICIT_MARKERS: dict[str, tuple[str, ...]] = {
    "task_type": ("task_type",),
    "task_prefix": ("task_prefix",),
    "priority": ("priority",),
    "dataset_paths": ("datasets.dataset_path",),
    "dataset_names": ("datasets.dataset_name",),
    "pipeline_stages": ("pipeline_stages",),
    "pipeline_mode": ("pipeline_stages",),
    "timeout_min": ("datasets.timeout_min",),
    "max_active_runs": ("max_active_runs",),
    "gpu_ids": ("gpu_ids",),
    "gpu_stage_memory_mb": ("gpu_stage_memory_mb",),
    "exclusive_gpu_stages": ("exclusive_gpu_stages",),
    "shared_gpu_stages": ("shared_gpu_stages",),
    "images": ("datasets.images",),
}


def _has_draft_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def merge_task_drafts(
    deterministic_draft: dict[str, Any] | None,
    model_draft: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge explicit user literals into the model's semantic task draft.

    The deterministic parser is authoritative only for fields it actually
    extracted.  It never supplies defaults or invents values; unresolved fields
    remain unresolved for the existing TaskPlanningService validation path.
    """

    deterministic = dict(deterministic_draft or {})
    merged = deepcopy(dict(model_draft or {}))
    explicit_fields = {str(item) for item in deterministic.get("explicit_fields") or []}
    for field in _DETERMINISTIC_DRAFT_FIELDS:
        value = deterministic.get(field)
        if not _has_draft_value(value):
            continue
        is_explicit = bool(explicit_fields.intersection(_FIELD_EXPLICIT_MARKERS.get(field, ())))
        # Explicit user literals override model output. Derived parser values
        # are only fallbacks and must not overwrite a model semantic value.
        if not is_explicit and _has_draft_value(merged.get(field)):
            continue
        if isinstance(value, dict):
            current = merged.get(field)
            current = dict(current) if isinstance(current, dict) else {}
            current.update(deepcopy(value))
            merged[field] = current
        else:
            merged[field] = deepcopy(value)

    merged_explicit_fields: list[str] = []
    for source in (merged.get("explicit_fields"), deterministic.get("explicit_fields")):
        for field in source or []:
            field = str(field)
            if field not in merged_explicit_fields:
                merged_explicit_fields.append(field)
    if merged_explicit_fields:
        merged["explicit_fields"] = merged_explicit_fields
    return merged


class TaskPlanningService:
    """Natural language -> TaskSpec -> existing platform validation -> YAML.

    This service has no submit/trigger dependency by design. Writing a preview YAML
    is allowed, but it never creates a DAG, DagRun, queue record, container or GPU
    reservation.
    """

    def __init__(
        self,
        defaults_path: str | Path | None = None,
        parser=None,
        scripts_dir: str | Path | None = None,
    ):
        self.defaults_loader = TaskPlanningDefaults(defaults_path)
        self.parser = parser or HeuristicTaskDraftParser()
        self.scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).resolve().parents[1] / "scripts"

    @classmethod
    def from_env(cls) -> "TaskPlanningService":
        return cls(defaults_path=os.environ.get("PLATFORM_TASK_PLANNING_DEFAULTS") or None)

    @staticmethod
    def _issue(code: str, path: str, message: str, severity: str = "error") -> ValidationIssue:
        return ValidationIssue(code=code, path=path, message=message, severity=severity)

    @staticmethod
    def _resolve_datasets(
        draft: dict[str, Any],
        defaults: dict[str, Any],
        stages: list[str],
        defaulted: list[str],
        unresolved: list[str],
    ) -> list[DatasetSpec]:
        dataset_defaults = deepcopy(defaults["dataset_defaults"])
        image_defaults = deepcopy(defaults["image_defaults"])
        explicit_images = dict(draft.get("images") or {})
        image_defaults.update(explicit_images)

        paths = list(draft.get("dataset_paths") or [])
        names = list(draft.get("dataset_names") or [])
        timeout = draft.get("timeout_min", dataset_defaults.get("timeout_min", 60))
        if "timeout_min" not in draft:
            defaulted.append("datasets.timeout_min")

        if not paths:
            unresolved.append("datasets.dataset_path")
            return []

        count = max(len(paths), len(names), 1)
        if len(paths) == 1 and count > 1:
            paths = paths * count
        if names and len(names) not in {1, len(paths)}:
            unresolved.append("datasets.dataset_name_alignment")
            return []
        if len(names) == 1 and len(paths) > 1:
            # Do not silently duplicate the same dataset_name.
            names = []

        result: list[DatasetSpec] = []
        for index, path in enumerate(paths):
            name = names[index] if index < len(names) else derive_dataset_name(path, index)
            if index >= len(names):
                defaulted.append(f"datasets[{index}].dataset_name")
            images: dict[str, str] = {}
            for stage in stages:
                if stage == "precheck":
                    continue
                value = str(image_defaults.get(stage) or "").strip()
                if value:
                    images[stage] = value
                    if stage not in explicit_images:
                        defaulted.append(f"datasets[{index}].image_{stage}")
            result.append(
                DatasetSpec(
                    dataset_name=name,
                    dataset_path=path,
                    tier=str(dataset_defaults.get("tier", "small")),
                    pool=str(dataset_defaults.get("pool", "default_pool")),
                    timeout_min=int(timeout),
                    images=images,
                    image_qc=str(dataset_defaults.get("image_qc", "")),
                )
            )
        return result

    @staticmethod
    def _to_platform_config(spec: TaskSpec) -> dict[str, Any]:
        config: dict[str, Any] = {
            "pipeline_stages": deepcopy(spec.pipeline_stages),
            "max_active_runs": spec.max_active_runs,
            "task_exclusive": spec.task_exclusive,
            "task_lock_wait_interval_sec": spec.task_lock_wait_interval_sec,
            "preempt_grace_timeout_min": spec.preempt_grace_timeout_min,
            "gpu_ids": spec.gpu_ids,
            "gpu_stages": spec.gpu_stages,
            "exclusive_gpu_stages": spec.exclusive_gpu_stages,
            "exclusive_gpu_idle_used_max_mb": spec.exclusive_gpu_idle_used_max_mb,
            "gpu_stage_memory_mb": deepcopy(spec.gpu_stage_memory_mb),
            "gpu_wait_interval_sec": spec.gpu_wait_interval_sec,
            "gpu_reservation_pending_sec": spec.gpu_reservation_pending_sec,
            "datasets": [],
        }
        if spec.task_type:
            config["task_type"] = spec.task_type
        if spec.priority is not None:
            config["priority"] = spec.priority
        for dataset in spec.datasets:
            row: dict[str, Any] = {
                "dataset_name": dataset.dataset_name,
                "tier": dataset.tier,
                "pool": dataset.pool,
                "dataset_path": dataset.dataset_path,
                "timeout_min": dataset.timeout_min,
            }
            for stage, image in dataset.images.items():
                row[f"image_{stage}"] = image
            if dataset.image_qc:
                row["image_qc"] = dataset.image_qc
            config["datasets"].append(row)
        return config

    def plan(self, user_text: str) -> TaskPlanningResult:
        text = (user_text or "").strip()
        if not text:
            return TaskPlanningResult(
                user_text=text,
                valid=False,
                unresolved_fields=["request"],
                issues=[self._issue("EMPTY_REQUEST", "request", "Task planning request must not be empty.")],
            )
        try:
            draft = self.parser.parse(text)
        except Exception as exc:
            return TaskPlanningResult(
                user_text=text,
                valid=False,
                issues=[self._issue("PLANNER_PARSE_FAILED", "planning", str(exc))],
            )
        return self.plan_from_draft(text, draft)

    def plan_from_draft(self, user_text: str, draft: dict[str, Any] | None) -> TaskPlanningResult:
        text = (user_text or "").strip()
        defaults_used: list[str] = []
        unresolved: list[str] = []
        issues: list[ValidationIssue] = []
        if not text:
            return TaskPlanningResult(
                user_text=text,
                valid=False,
                unresolved_fields=["request"],
                issues=[self._issue("EMPTY_REQUEST", "request", "Task planning request must not be empty.")],
            )
        try:
            deterministic_draft = HeuristicTaskDraftParser().parse(text)
        except Exception:
            # The explicit parser is deliberately best-effort.  Its failure must
            # leave the model draft untouched and let normal validation report
            # unresolved fields.
            deterministic_draft = {}
        model_draft = dict(draft or {})
        deterministic_explicit_fields = set(deterministic_draft.get("explicit_fields") or [])
        derived_prefix_from_type = (
            _has_draft_value(deterministic_draft.get("task_prefix"))
            and "task_prefix" not in deterministic_explicit_fields
            and _has_draft_value(deterministic_draft.get("task_type"))
            and not _has_draft_value(model_draft.get("task_prefix"))
        )
        draft = merge_task_drafts(deterministic_draft, model_draft)
        try:
            defaults = self.defaults_loader.load()
        except Exception as exc:
            return TaskPlanningResult(
                user_text=text,
                valid=False,
                issues=[self._issue("PLANNER_INIT_FAILED", "planning", str(exc))],
            )

        task_defaults = deepcopy(defaults["task_defaults"])
        explicit_fields = list(dict.fromkeys(draft.get("explicit_fields") or []))

        task_prefix = str(draft.get("task_prefix") or "").strip()
        if derived_prefix_from_type and task_prefix:
            defaults_used.append("task_prefix_from_task_type")
        if not task_prefix and str(draft.get("task_type") or "").strip():
            task_prefix = str(draft.get("task_type")).strip().lower()
            defaults_used.append("task_prefix_from_task_type")
        if not task_prefix:
            unresolved.append("task_prefix")
            task_prefix = "task"
        else:
            try:
                validate_task_name(task_prefix)
                if len(task_prefix) > MAX_TASK_PREFIX_LENGTH:
                    raise TaskConfigError(
                        f"task prefix must not exceed {MAX_TASK_PREFIX_LENGTH} characters: {task_prefix!r}"
                    )
            except TaskConfigError as exc:
                issues.append(self._issue("INVALID_TASK_PREFIX", "task_prefix", str(exc)))

        pipeline = draft.get("pipeline_stages")
        if not pipeline:
            pipeline = deepcopy(task_defaults.get("pipeline_stages") or [])
            defaults_used.append("pipeline_stages")

        # Build a temporary config only to normalize stage syntax here. The full
        # deterministic validator is called again after all fields are assembled.
        try:
            stage_groups = normalize_pipeline_stages({"pipeline_stages": pipeline})
            stages = flatten_stage_groups(stage_groups)
            pipeline = [group[0] if len(group) == 1 else group for group in stage_groups]
        except TaskConfigError as exc:
            stages = []
            issues.append(self._issue("INVALID_PIPELINE", "pipeline_stages", str(exc)))

        max_active_runs = int(draft.get("max_active_runs", task_defaults.get("max_active_runs", 5)))
        if "max_active_runs" not in draft:
            defaults_used.append("max_active_runs")

        gpu_defaults = deepcopy(task_defaults.get("gpu") or {})
        gpu_stages_list = [stage for stage in stages if stage in set(gpu_defaults.get("gpu_stages") or [])]
        # If an explicitly mentioned memory value adds a GPU stage, keep it.
        for stage in (draft.get("gpu_stage_memory_mb") or {}):
            if stage in stages and stage not in gpu_stages_list:
                gpu_stages_list.append(stage)

        exclusive = list(gpu_defaults.get("exclusive_gpu_stages") or [])
        if "exclusive_gpu_stages" in draft:
            exclusive = list(draft.get("exclusive_gpu_stages") or [])
        shared = set(draft.get("shared_gpu_stages") or [])
        exclusive = [stage for stage in exclusive if stage in gpu_stages_list and stage not in shared]

        memory = deepcopy(gpu_defaults.get("gpu_stage_memory_mb") or {})
        memory.update(draft.get("gpu_stage_memory_mb") or {})
        memory = {stage: int(memory[stage]) for stage in gpu_stages_list if stage in memory}

        if gpu_stages_list:
            if "gpu_ids" not in draft:
                defaults_used.append("gpu_ids")
            if "gpu_stage_memory_mb" not in draft:
                defaults_used.append("gpu_stage_memory_mb")
            if "exclusive_gpu_stages" not in draft and "shared_gpu_stages" not in draft:
                defaults_used.append("exclusive_gpu_stages")

        datasets = self._resolve_datasets(draft, defaults, stages, defaults_used, unresolved)

        try:
            spec = TaskSpec(
                task_prefix=task_prefix,
                task_type=str(draft.get("task_type") or task_defaults.get("task_type") or ""),
                priority=draft.get("priority"),
                pipeline_stages=pipeline,
                max_active_runs=max_active_runs,
                task_exclusive=bool(task_defaults.get("task_exclusive", True)),
                task_lock_wait_interval_sec=int(task_defaults.get("task_lock_wait_interval_sec", 10)),
                preempt_grace_timeout_min=int(task_defaults.get("preempt_grace_timeout_min", 60)),
                gpu_ids=str(draft.get("gpu_ids") or gpu_defaults.get("gpu_ids") or "") if gpu_stages_list else "",
                gpu_stages=",".join(gpu_stages_list),
                exclusive_gpu_stages=",".join(exclusive),
                exclusive_gpu_idle_used_max_mb=int(gpu_defaults.get("exclusive_gpu_idle_used_max_mb", 512)),
                gpu_stage_memory_mb=memory,
                gpu_wait_interval_sec=int(gpu_defaults.get("gpu_wait_interval_sec", 10)),
                gpu_reservation_pending_sec=int(gpu_defaults.get("gpu_reservation_pending_sec", 60)),
                datasets=datasets,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return TaskPlanningResult(
                user_text=text,
                valid=False,
                defaults_used=list(dict.fromkeys(defaults_used)),
                explicit_fields=explicit_fields,
                unresolved_fields=list(dict.fromkeys(unresolved)),
                issues=[self._issue("TASK_SPEC_INVALID", "task_spec", str(exc))] + issues,
            )

        config = self._to_platform_config(spec)
        resolved_priority = None
        priority_source = ""
        try:
            priority_config = normalize_task_priority_config(config)
            resolved_priority = int(priority_config["priority"])
            priority_source = str(priority_config["priority_source"])
        except TaskConfigError as exc:
            issues.append(self._issue("INVALID_PRIORITY", "priority", str(exc)))

        if unresolved:
            for field in list(dict.fromkeys(unresolved)):
                issues.append(
                    self._issue(
                        "UNRESOLVED_FIELD",
                        field,
                        f"Required planning field is unresolved: {field}",
                    )
                )

        if not unresolved and not any(item.severity == "error" for item in issues):
            try:
                validate_config(config, scripts_dir=self.scripts_dir)
            except TaskConfigError as exc:
                issues.append(self._issue("PLATFORM_VALIDATION_FAILED", "config", str(exc)))

        yaml_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        valid = not any(item.severity == "error" for item in issues)
        return TaskPlanningResult(
            user_text=text,
            valid=valid,
            task_spec=spec,
            config=config,
            yaml_text=yaml_text,
            resolved_priority=resolved_priority,
            priority_source=priority_source,
            defaults_used=list(dict.fromkeys(defaults_used)),
            explicit_fields=explicit_fields,
            unresolved_fields=list(dict.fromkeys(unresolved)),
            issues=issues,
        )

    @staticmethod
    def write_yaml(result: TaskPlanningResult, output_path: str | Path) -> Path:
        if not result.valid:
            raise TaskConfigError("Refusing to write invalid TaskSpec YAML")
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp.write_text(result.yaml_text, encoding="utf-8")
        os.replace(temp, path)
        return path
