"""Shared routing language used by every structured Planner provider."""

EVIDENCE_ROUTING_CONTRACT = """Evidence routing taxonomy (classify the user's goal and requested evidence, not keyword overlap):
- STATIC_KNOWLEDGE: definitions, architecture, mechanisms, policies, rules, runbooks or recovery explanations, when the user is not asking about current runtime state. Use intent=platform_knowledge and call search_knowledge when it is available. Do not use get_gpu_pool to explain a platform concept.
- LIVE_GPU_STATE: current GPU memory, availability, active reservations, or why a GPU cannot be allocated now. Use intent=gpu_diagnosis and get_gpu_pool. search_knowledge may supplement an explicitly requested rule explanation but never replaces live GPU evidence.
- LIVE_TASK_STATE: the current status, queue position or recent state of a concrete named task without a failure/stuck diagnosis. Use intent=task_status and get_task_detail; get_queue_state may supplement queue context. Do not turn a plain status question into task_diagnosis.
- NAMED_TASK_DIAGNOSIS: a stuck, failed, pending or non-running concrete task identified by an actual task name. Use intent=task_diagnosis and diagnose_task(task_name=...). Never invent a task_name from a stage, component or dataset label.
- TASK_PLANNING: generate or validate a task configuration/YAML/pipeline without an explicit submit, trigger, start or execute request. Use intent=task_planning and tool_calls=[]; deterministic TaskPlanningService handles the draft.
- WRITE_OPERATION: explicit mutation or execution such as submit, trigger, start, execute, stop, delete, resume or priority change. Use the matching write intent and frozen write_action, never a write tool in tool_calls; the workflow preserves read evidence, impact analysis, HITL, precondition, mutation and verification gates.
- NO_TOOL: greetings and ordinary conversation with no platform fact use tool_calls=[].
Distinguish evidence goals even when words overlap: 'GPU Reservation 是什么？' is STATIC_KNOWLEDGE and uses search_knowledge; '现在 GPU0 上有哪些 Reservation？' is LIVE_GPU_STATE and uses get_gpu_pool."""
