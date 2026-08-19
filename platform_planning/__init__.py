"""Natural-language task planning for the offline processing platform.

V0.6 deliberately stops at validated YAML generation. It never submits, triggers,
or mutates an Airflow task.
"""

from .models import DatasetSpec, TaskPlanningResult, TaskSpec, ValidationIssue
from .service import TaskPlanningService

__all__ = [
    "DatasetSpec",
    "TaskSpec",
    "ValidationIssue",
    "TaskPlanningResult",
    "TaskPlanningService",
]
