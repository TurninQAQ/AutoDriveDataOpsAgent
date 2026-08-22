# AutoDriveDataOpsAgent V2 — Phase A/B

This directory is a self-contained V2 package for the first explicit READ-only
Agent loop. It has no runtime dependency on `deploy_ci_cloud_agent`.

The host enters through:

```python
from deploy_ci_cloud_agentv2 import build_system_context, invoke

result = await invoke(
    "task_A 现在什么状态？",
    thread_id="thread-1",
    system_context=build_system_context(),
)
```

The visible LangGraph is:

```text
START -> agent -> read_executor -> agent
agent -> response_completion_gate -> agent / END
runtime terminal -> END
```

The default provider and `InMemoryReadFacade` are offline-safe fixtures. A host
can inject its own V2-local `AgentProvider` and `ReadFacade` without importing
V1 modules. Phase B contains no WRITE execution, HITL interrupt, AUTO path, or
autonomy policy.

READ transport success is kept separate from qualified evidence. Entity-bound
tools compare requested and observed identity; missing or conflicting identity,
placeholder state, empty knowledge, and empty diagnosis fail closed. The
completion gate also requires a `FinalCandidate` to reference every declared
goal. Invalid READ proposals become bounded guard observations and return to
the Agent. Runtime READ retries are side-effect-free and bounded per tool call.

Run the local tests from the repository root:

```bash
PYTHONPATH=/home/ubuntu/project/AutoDriveDataOpsAgent \
  .venv/bin/pytest -q deploy_ci_cloud_agentv2/tests
```
