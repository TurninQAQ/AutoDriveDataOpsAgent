from __future__ import annotations

from typing import Any

from .facade import PlatformMCPFacade, build_default_facade


READ_ONLY_TOOL_NAMES = (
    "get_platform_health",
    "list_tasks",
    "get_task_detail",
    "get_queue_state",
    "get_gpu_pool",
    "inspect_task_containers",
    "get_stage_logs",
    "diagnose_task",
    "search_knowledge",
)

WRITE_PREP_TOOL_NAMES = (
    "get_write_precondition",
    "validate_task_spec",
    "get_action_verification_snapshot",
)

WRITE_TOOL_NAMES = (
    "submit_task",
    "resume_task",
    "set_task_priority",
    "stop_task",
    "delete_task",
)

ALL_TOOL_NAMES = READ_ONLY_TOOL_NAMES + WRITE_PREP_TOOL_NAMES + WRITE_TOOL_NAMES

MCP_TOOL_DESCRIPTIONS = {
    "get_gpu_pool": (
        "Inspect current/live GPU runtime state, including device memory, "
        "availability and active reservations. Use for what GPU resources or "
        "reservations exist now, or current GPU allocation problems. Do not use "
        "it to explain platform concepts, architecture or reservation rules."
    ),
    "search_knowledge": (
        "Search static platform documentation and runbooks for definitions, "
        "architecture, mechanisms, policies and operating rules such as GPU "
        "Reservation, soft preemption and recovery. Use for what-is, how-it-works "
        "and platform-rule questions. It does not return current runtime state."
    ),
}


def build_mcp_server(facade: PlatformMCPFacade | None = None, include_write_tools: bool = False):
    """Build the official MCP Python SDK v2 server.

    V0.8 exposes read-only, guarded write and internal verification tools. Write tools still enforce
    deterministic validation and optimistic preconditions inside Platform Core;
    Agent approval alone is never treated as authorization to bypass those checks.
    """
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised in runtime environment
        raise RuntimeError(
            "MCP Python SDK is not installed. Install requirements-mcp.txt first."
        ) from exc

    facade = facade or build_default_facade()
    mcp = MCPServer(
        "autodrive-dataops-platform",
        instructions=(
            "Domain tools for the automatic-driving offline data processing platform. "
            "Read tools inspect Airflow, queue, Docker and GPU evidence. State-changing "
            "tools require an approval-created precondition and revalidate it in Platform Core."
        ),
    )

    @mcp.tool()
    def get_platform_health() -> dict[str, Any]:
        """Get platform component health and resource availability."""
        return facade.get_platform_health()

    @mcp.tool()
    def list_tasks(limit: int = 100) -> dict[str, Any]:
        """List generated business tasks with priority and queue information."""
        return facade.list_tasks(limit=limit)

    @mcp.tool()
    def get_task_detail(
        task_name: str, include_airflow_runs: bool = True, run_limit: int = 20
    ) -> dict[str, Any]:
        """Get one business task's config, datasets, queue status and recent DagRuns."""
        return facade.get_task_detail(task_name, include_airflow_runs, run_limit)

    @mcp.tool()
    def get_queue_state(task_name: str = "") -> dict[str, Any]:
        """Get the global priority queue or one task's queue position."""
        return facade.get_queue_state(task_name)

    def get_gpu_pool(cleanup_dead: bool = True) -> dict[str, Any]:
        return facade.get_gpu_pool(cleanup_dead=cleanup_dead)

    get_gpu_pool.__doc__ = MCP_TOOL_DESCRIPTIONS["get_gpu_pool"]
    get_gpu_pool = mcp.tool()(get_gpu_pool)

    @mcp.tool()
    def inspect_task_containers(
        task_name: str, datasets: list[str] | None = None
    ) -> dict[str, Any]:
        """Inspect running Docker containers belonging to a task/dataset."""
        return facade.inspect_task_containers(task_name, datasets)

    @mcp.tool()
    def get_stage_logs(
        task_name: str,
        dataset_name: str = "",
        stage: str = "",
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        """Get recent Airflow logs for a task's failed/running or selected Stage tasks."""
        return facade.get_stage_logs(task_name, dataset_name, stage, tail_lines)

    @mcp.tool()
    def diagnose_task(task_name: str, dataset_name: str = "") -> dict[str, Any]:
        """Aggregate queue, Airflow, Docker and GPU evidence without LLM inference."""
        return facade.diagnose_task(task_name, dataset_name)

    if getattr(facade, "knowledge_service", None) is not None:
        def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
            return facade.search_knowledge(query, top_k)

        search_knowledge.__doc__ = MCP_TOOL_DESCRIPTIONS["search_knowledge"]
        search_knowledge = mcp.tool()(search_knowledge)

    if include_write_tools:
        @mcp.tool()
        def get_write_precondition(task_name: str = "") -> dict[str, Any]:
            """Capture the exact state fingerprint used when a write action is approved."""
            return facade.get_write_precondition(task_name)

        @mcp.tool()
        def validate_task_spec(task_prefix: str, config: dict[str, Any]) -> dict[str, Any]:
            """Revalidate a V0.6 TaskSpec/YAML without mutating the platform."""
            return facade.validate_task_spec(task_prefix, config)

        @mcp.tool()
        def get_action_verification_snapshot(
            task_name: str, datasets: list[str] | None = None, airflow_limit: int = 100
        ) -> dict[str, Any]:
            """Collect deterministic post-action task/queue/Docker/GPU/Airflow evidence."""
            return facade.get_action_verification_snapshot(task_name, datasets, airflow_limit)

        @mcp.tool()
        def submit_task(
            task_prefix: str, config: dict[str, Any], precondition: dict[str, Any]
        ) -> dict[str, Any]:
            """Submit a previously validated task after explicit HITL approval."""
            return facade.submit_task(task_prefix, config, precondition)

        @mcp.tool()
        def resume_task(
            task_name: str,
            datasets: list[str] | None,
            precondition: dict[str, Any],
        ) -> dict[str, Any]:
            """Resume failed/selected datasets after explicit HITL approval."""
            return facade.resume_task(task_name, datasets, precondition)

        @mcp.tool()
        def set_task_priority(
            task_name: str, priority: int, precondition: dict[str, Any]
        ) -> dict[str, Any]:
            """Change business task priority after impact analysis and HITL approval."""
            return facade.set_task_priority(task_name, priority, precondition)

        @mcp.tool()
        def stop_task(
            task_name: str,
            datasets: list[str] | None,
            precondition: dict[str, Any],
        ) -> dict[str, Any]:
            """Stop all or selected datasets after explicit HITL approval."""
            return facade.stop_task(task_name, datasets, precondition)

        @mcp.tool()
        def delete_task(task_name: str, precondition: dict[str, Any]) -> dict[str, Any]:
            """Delete a generated business task after strong HITL approval."""
            return facade.delete_task(task_name, precondition)

    return mcp


def main() -> None:
    import sys

    try:
        # Keep the stdio MCP entrypoint aligned with the Agent runtime's configured
        # embedding/index settings without importing Agent modules at module load.
        from platform_agent.runtime import build_agent_knowledge_service
        from platform_agent.settings import AgentSettings
        from platform_core.settings import PlatformSettings

        platform_settings = PlatformSettings.from_env()
        agent_settings = AgentSettings.from_env(platform_settings)
        facade = build_default_facade(
            platform_settings,
            knowledge_service=build_agent_knowledge_service(agent_settings),
        )
        server = build_mcp_server(facade, include_write_tools=True)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
