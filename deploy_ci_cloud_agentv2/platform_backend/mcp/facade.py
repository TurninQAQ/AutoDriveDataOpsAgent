from __future__ import annotations

import contextlib
import io
import sys
from typing import TYPE_CHECKING, Any

from deploy_ci_cloud_agentv2.platform_backend.core.gateways.airflow_read import AirflowReadGateway
from deploy_ci_cloud_agentv2.platform_backend.core.gateways.docker import DockerGateway
from deploy_ci_cloud_agentv2.platform_backend.core.gateways.gpu_runtime import create_gpu_runtime_from_env
from deploy_ci_cloud_agentv2.platform_backend.core.gateways.legacy_mutation import LegacyMutationGateway
from deploy_ci_cloud_agentv2.platform_backend.core.errors import TaskConfigError
from deploy_ci_cloud_agentv2.platform_backend.core.services.airflow_read_service import AirflowReadService
from deploy_ci_cloud_agentv2.platform_backend.core.services.diagnosis_service import DiagnosisService
from deploy_ci_cloud_agentv2.platform_backend.core.services.docker_service import DockerService
from deploy_ci_cloud_agentv2.platform_backend.core.services.gpu_service import GPUService
from deploy_ci_cloud_agentv2.platform_backend.core.services.health_service import HealthService
from deploy_ci_cloud_agentv2.platform_backend.core.services.queue_service import QueueService
from deploy_ci_cloud_agentv2.platform_backend.core.services.precondition_service import PreconditionService
from deploy_ci_cloud_agentv2.platform_backend.core.services.mutation_service import PlatformMutationService
from deploy_ci_cloud_agentv2.platform_backend.core.services.task_query_service import TaskQueryService
from deploy_ci_cloud_agentv2.platform_backend.core.services.verification_service import ActionVerificationSnapshotService
from deploy_ci_cloud_agentv2.platform_backend.core.settings import PlatformSettings
from deploy_ci_cloud_agentv2.platform_backend.core.task_store import dataset_map, load_task_config, task_paths

if TYPE_CHECKING:
    from deploy_ci_cloud_agentv2.platform_backend.rag.service import KnowledgeService


class PlatformMCPFacade:
    """Transport-independent implementation of the V0.3 read-only tool surface."""

    def __init__(
        self,
        settings: PlatformSettings,
        task_query_service: TaskQueryService,
        queue_service: QueueService,
        gpu_service: GPUService,
        docker_service: DockerService,
        airflow_service: AirflowReadService,
        diagnosis_service: DiagnosisService,
        health_service: HealthService,
        mutation_service: PlatformMutationService | None = None,
        verification_service: ActionVerificationSnapshotService | None = None,
        knowledge_service: KnowledgeService | None = None,
    ):
        self.settings = settings
        self.task_query_service = task_query_service
        self.queue_service = queue_service
        self.gpu_service = gpu_service
        self.docker_service = docker_service
        self.airflow_service = airflow_service
        self.diagnosis_service = diagnosis_service
        self.health_service = health_service
        self.mutation_service = mutation_service
        self.verification_service = verification_service
        self.knowledge_service = knowledge_service

    @contextlib.contextmanager
    def _stdio_safe(self):
        """Redirect legacy platform prints away from MCP stdio protocol stdout."""
        with contextlib.redirect_stdout(sys.stderr):
            yield

    def get_platform_health(self) -> dict[str, Any]:
        """Return Airflow, queue, Docker, task-config and GPU health evidence."""
        with self._stdio_safe():
            return self.health_service.snapshot()

    def list_tasks(self, limit: int = 100) -> dict[str, Any]:
        """List generated business tasks and their queue/priority summary."""
        with self._stdio_safe():
            tasks = self.task_query_service.list_tasks(limit=limit)
        return {"count": len(tasks), "tasks": tasks}

    def get_task_detail(
        self, task_name: str, include_airflow_runs: bool = True, run_limit: int = 20
    ) -> dict[str, Any]:
        """Return one task's configuration, datasets, queue state and recent DagRuns."""
        with self._stdio_safe():
            try:
                detail = self.task_query_service.get_task_detail(task_name)
            except TaskConfigError as exc:
                if str(exc).startswith("Task config not found:"):
                    return {"status": "NOT_FOUND", "task_name": task_name, "exists": False}
                raise
            detail["exists"] = True
            if include_airflow_runs:
                try:
                    detail["airflow_runs"] = self.airflow_service.runs(
                        detail["dag_id"], limit=max(1, min(int(run_limit), 100))
                    )
                    detail["airflow_error"] = None
                except Exception as exc:
                    detail["airflow_runs"] = []
                    detail["airflow_error"] = str(exc)
            detail["state"] = _task_state_from_runs(detail.get("airflow_runs"))
        return detail

    def get_queue_state(self, task_name: str = "") -> dict[str, Any]:
        """Return the global business-task queue or one task's queue position."""
        with self._stdio_safe():
            if task_name:
                return self.queue_service.task_status(task_name)
            return self.queue_service.snapshot()

    def get_gpu_pool(self, cleanup_dead: bool = True) -> dict[str, Any]:
        """Return GPU memory plus current platform Reservation entries."""
        with self._stdio_safe():
            devices = self.gpu_service.device_snapshot()
            reservations = []
            for lock_name, token, item in self.gpu_service.reservations(
                cleanup_dead=cleanup_dead
            ):
                gpu_id = lock_name.removeprefix("gpu_").removesuffix(".lock")
                reservations.append(
                    {
                        "gpu_id": gpu_id,
                        "token": token,
                        "pid": item.get("pid"),
                        "task_name": item.get("task_name"),
                        "dataset_name": item.get("dataset_name"),
                        "stage": item.get("stage"),
                        "required_mb": item.get("required_mb"),
                        "exclusive": bool(item.get("exclusive", False)),
                    }
                )
        return {"devices": devices, "reservations": reservations}

    def inspect_task_containers(
        self, task_name: str, datasets: list[str] | None = None
    ) -> dict[str, Any]:
        """Inspect Docker containers that belong to a task and optional datasets."""
        with self._stdio_safe():
            _, config = load_task_config(task_name, self.settings.task_config_root)
            available = list(dataset_map(config))
            selected = list(datasets or available)
            unknown = [name for name in selected if name not in available]
            if unknown:
                raise ValueError("Unknown dataset_name: " + ",".join(unknown))
            containers = self.docker_service.matching_containers(task_name, config, selected)
            return {
                "task_name": task_name,
                "datasets": selected,
                "containers": [
                    self.diagnosis_service._container_summary(item) for item in containers
                ],
            }

    def get_stage_logs(
        self,
        task_name: str,
        dataset_name: str = "",
        stage: str = "",
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        """Return recent Airflow log tails for a task/dataset/stage."""
        with self._stdio_safe():
            detail = self.task_query_service.get_task_detail(task_name)
            dataset = dataset_name or None
            run = self.airflow_service.latest_run(detail["dag_id"], dataset)
            if not run:
                return {
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "stage": stage,
                    "logs": [],
                    "message": "No matching DagRun found",
                }
            run_id = str(run.get("dag_run_id") or run.get("run_id") or "")
            if not run_id:
                raise RuntimeError("Matching DagRun did not contain run id")
            instances = self.airflow_service.task_instances(detail["dag_id"], run_id)
            stage = stage.strip().lower()
            candidates = []
            for item in instances:
                task_id = str(item.get("task_id") or "")
                if stage and task_id not in {f"run_{stage}", f"validate_{stage}"}:
                    continue
                state = str(item.get("state") or "").lower()
                if not stage and state not in {"failed", "upstream_failed", "running"}:
                    continue
                candidates.append(item)
            if not candidates and stage:
                candidates = [
                    item for item in instances
                    if str(item.get("task_id") or "") in {f"run_{stage}", f"validate_{stage}"}
                ]
            logs = []
            for item in candidates[:4]:
                task_id = str(item.get("task_id") or "")
                try_number = int(item.get("try_number") or 1)
                map_index = int(item.get("map_index") if item.get("map_index") is not None else -1)
                try:
                    payload = self.airflow_service.task_log(
                        detail["dag_id"], run_id, task_id,
                        try_number=try_number,
                        map_index=map_index,
                        tail_lines=tail_lines,
                    )
                    payload["state"] = item.get("state")
                    logs.append(payload)
                except Exception as exc:
                    logs.append({"task_id": task_id, "state": item.get("state"), "error": str(exc)})
            return {
                "task_name": task_name,
                "dataset_name": dataset_name,
                "stage": stage,
                "run_id": run_id,
                "logs": logs,
            }

    def diagnose_task(self, task_name: str, dataset_name: str = "") -> dict[str, Any]:
        """Aggregate queue, Airflow, Docker and GPU evidence for a task."""
        with self._stdio_safe():
            _, config = load_task_config(task_name, self.settings.task_config_root)
            available = list(dataset_map(config))
            if dataset_name and dataset_name not in available:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")
            selected = [dataset_name] if dataset_name else available
            dag_id = str(task_paths(task_name, self.settings.dags_dir, self.settings.task_config_root)["dag_id"])
            result = self.diagnosis_service.inspect_task(
                task_name,
                config,
                selected,
                dag_id=dag_id,
                dataset_name=dataset_name or None,
            )
            result["dag_id"] = dag_id
            return result

    def search_knowledge(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the static platform knowledge index without mutating state."""
        if self.knowledge_service is None:
            raise RuntimeError("Knowledge search service is not configured")
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        limit = max(1, min(int(top_k), 100))
        with self._stdio_safe():
            result = self.knowledge_service.search(query, top_k=limit)
        rows = []
        for rank, item in enumerate(result.results, start=1):
            rows.append(
                {
                    "rank": rank,
                    "source": item.citation,
                    "source_path": item.source_path,
                    "chunk_id": item.chunk_id,
                    "title": item.title,
                    "section": item.section,
                    "content": item.content,
                    "score": item.score,
                    "lexical_score": item.lexical_score,
                    "vector_score": item.vector_score,
                    "metadata": item.metadata,
                }
            )
        return {
            "query": result.query,
            "top_k": limit,
            "count": len(rows),
            "results": rows,
            "index_stats": result.index_stats.model_dump(mode="json") if result.index_stats else None,
        }

    def _require_mutation_service(self) -> PlatformMutationService:
        if self.mutation_service is None:
            raise RuntimeError("Platform mutation service is not configured")
        return self.mutation_service

    def get_action_verification_snapshot(
        self, task_name: str, datasets: list[str] | None = None, airflow_limit: int = 100
    ) -> dict[str, Any]:
        """Collect post-mutation evidence even if task config has already been deleted."""
        if self.verification_service is None:
            raise RuntimeError("Action verification service is not configured")
        with self._stdio_safe():
            return self.verification_service.snapshot(task_name, datasets=datasets, airflow_limit=airflow_limit)

    def get_write_precondition(self, task_name: str = "") -> dict[str, Any]:
        """Capture an optimistic state fingerprint used by approved write actions."""
        with self._stdio_safe():
            return self._require_mutation_service().capture_precondition(task_name)

    def validate_task_spec(self, task_prefix: str, config: dict[str, Any]) -> dict[str, Any]:
        """Revalidate a planned TaskSpec/YAML at the platform mutation boundary."""
        with self._stdio_safe():
            return self._require_mutation_service().validate_task_spec(task_prefix, config)

    def submit_task(
        self, task_prefix: str, config: dict[str, Any], precondition: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit a validated task config after an approval-time precondition check."""
        with self._stdio_safe():
            return self._require_mutation_service().submit_task(task_prefix, config, precondition)

    def resume_task(
        self, task_name: str, datasets: list[str] | None, precondition: dict[str, Any]
    ) -> dict[str, Any]:
        """Resume failed or selected datasets after approval and precondition validation."""
        with self._stdio_safe():
            return self._require_mutation_service().resume_task(task_name, datasets, precondition)

    def set_task_priority(
        self, task_name: str, priority: int, precondition: dict[str, Any]
    ) -> dict[str, Any]:
        """Change business task priority after approval and precondition validation."""
        with self._stdio_safe():
            return self._require_mutation_service().set_task_priority(task_name, priority, precondition)

    def stop_task(
        self, task_name: str, datasets: list[str] | None, precondition: dict[str, Any]
    ) -> dict[str, Any]:
        """Stop a task or selected datasets after explicit approval."""
        with self._stdio_safe():
            return self._require_mutation_service().stop_task(task_name, datasets, precondition)

    def delete_task(self, task_name: str, precondition: dict[str, Any]) -> dict[str, Any]:
        """Delete a generated task after strong approval and precondition validation."""
        with self._stdio_safe():
            return self._require_mutation_service().delete_task(task_name, precondition)


def build_default_facade(
    settings: PlatformSettings | None = None,
    knowledge_service: KnowledgeService | None = None,
) -> PlatformMCPFacade:
    settings = settings or PlatformSettings.from_env()
    queue_service = QueueService(settings.queue_file)
    task_query = TaskQueryService(
        settings.task_config_root, settings.dags_dir, queue_service=queue_service
    )
    gpu_runtime = create_gpu_runtime_from_env()
    gpu_service = GPUService(runtime=gpu_runtime, lock_dir=settings.gpu_lock_dir)
    docker_service = DockerService(DockerGateway())
    airflow_gateway = AirflowReadGateway(
        settings.airflow_api_base,
        user=settings.airflow_api_user,
        password=settings.airflow_api_password,
        token=settings.airflow_api_token,
        password_file=settings.airflow_password_file,
        timeout_sec=settings.api_timeout_sec,
    )
    airflow_service = AirflowReadService(airflow_gateway)
    diagnosis = DiagnosisService(
        queue_service,
        docker_service=docker_service,
        gpu_service=gpu_service,
        airflow_service=airflow_service,
    )
    health = HealthService(
        queue_service,
        airflow_service=airflow_service,
        gpu_service=gpu_service,
        task_config_root=settings.task_config_root,
    )
    preconditions = PreconditionService(
        queue_service, settings.dags_dir, settings.task_config_root
    )
    mutation = PlatformMutationService(
        LegacyMutationGateway(settings), preconditions
    )
    verification = ActionVerificationSnapshotService(
        task_config_root=settings.task_config_root, dags_dir=settings.dags_dir,
        queue_service=queue_service, docker_gateway=docker_service.gateway,
        gpu_service=gpu_service, airflow_service=airflow_service,
    )
    return PlatformMCPFacade(
        settings,
        task_query,
        queue_service,
        gpu_service,
        docker_service,
        airflow_service,
        diagnosis,
        health,
        mutation,
        verification,
        knowledge_service,
    )


def _task_state_from_runs(runs: Any) -> str:
    """Normalize platform run state into the V2 TaskState vocabulary."""

    if not isinstance(runs, list) or not runs:
        return "SUBMITTED"
    state = str((runs[0] or {}).get("state") or "").strip().lower()
    return {
        "queued": "QUEUED",
        "scheduled": "QUEUED",
        "running": "RUNNING",
        "up_for_retry": "RUNNING",
        "up_for_reschedule": "RUNNING",
        "deferred": "RUNNING",
        "success": "SUCCEEDED",
        "failed": "FAILED",
        "upstream_failed": "FAILED",
        "canceled": "CANCELLED",
        "cancelled": "CANCELLED",
    }.get(state, "SUBMITTED")
