from __future__ import annotations

import time
from typing import Any, TypedDict

from platform_mcp.server import READ_ONLY_TOOL_NAMES

from .actions import WRITE_INTENT_TO_TOOL, WriteActionCoordinator
from .memory import ConversationStore
from .models import (
    AgentIntent,
    AgentPlan,
    AgentResponse,
    ConversationTurn,
    KnowledgeObservation,
    ToolObservation,
)
from .policy import AgentPolicyEngine


WRITE_INTENTS = frozenset(WRITE_INTENT_TO_TOOL)


class AgentGraphState(TypedDict, total=False):
    user_text: str
    thread_id: str
    history: list[dict[str, Any]]
    plan: dict[str, Any]
    knowledge: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    response: dict[str, Any]
    task_planning: dict[str, Any]
    trace_id: str


class ReadOnlyAgentNodes:
    """Agent nodes shared by sequential and LangGraph runtimes.

    The historical class name is retained for compatibility. In V0.8 normal model
    tool calls are still read-only; state-changing calls can only be executed by the
    separate WriteActionCoordinator after persisted HITL approval.
    """

    def __init__(
        self,
        model,
        tool_client,
        policy: AgentPolicyEngine,
        knowledge_retriever=None,
        knowledge_top_k: int = 5,
        task_planning_service=None,
        action_coordinator: WriteActionCoordinator | None = None,
        trace_recorder=None,
    ):
        self.model = model
        self.tool_client = tool_client
        self.policy = policy
        self.knowledge_retriever = knowledge_retriever
        self.knowledge_top_k = max(1, knowledge_top_k)
        self.task_planning_service = task_planning_service
        self.action_coordinator = action_coordinator
        self.trace_recorder = trace_recorder
        self._tool_descriptions: list[dict[str, Any]] | None = None

    @staticmethod
    def _history(state: AgentGraphState) -> list[ConversationTurn]:
        return [ConversationTurn.model_validate(item) for item in state.get("history", [])]

    async def _tools(self) -> list[dict[str, Any]]:
        if self._tool_descriptions is None:
            tools = await self.tool_client.describe_tools()
            self._tool_descriptions = [item for item in tools if item.get("name") in READ_ONLY_TOOL_NAMES]
        return list(self._tool_descriptions)

    def validate_plan(self, plan: AgentPlan) -> AgentPlan:
        self.policy.validate_tool_count(len(plan.tool_calls))
        for call in plan.tool_calls:
            self.policy.validate_tool_name(call.name)
        if plan.intent == AgentIntent.UNSUPPORTED_WRITE and plan.tool_calls:
            raise PermissionError("unsupported_write plan must not execute tools")
        if plan.intent in WRITE_INTENTS:
            tool_name = WRITE_INTENT_TO_TOOL[plan.intent]
            self.policy.validate_write_tool(tool_name)
            if plan.write_action is None:
                raise PermissionError(f"Write intent {plan.intent.value} must provide frozen write_action arguments")
        elif plan.write_action:
            raise PermissionError("Non-write Agent intent must not carry write_action")
        return plan

    async def plan(self, state: AgentGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        user_text = state.get("user_text", "")
        # V0.8 no longer blanket-blocks supported mutations. The model may only
        # create a write intent/frozen arguments; policy still prevents write tools
        # from entering normal tool_calls before approval.
        if not getattr(self.policy, "supports_writes", False) and self.policy.is_write_request(user_text):
            plan = AgentPlan(
                intent=AgentIntent.UNSUPPORTED_WRITE,
                tool_calls=[],
                decision_summary="Read-only compatibility policy blocked mutation before model/tool execution.",
            )
            self.validate_plan(plan)
            if self.trace_recorder is not None:
                self.trace_recorder.record(
                    state.get("trace_id", ""), "plan", "agent_plan", duration_ms=(time.perf_counter() - started) * 1000,
                    data={"intent": plan.intent.value, "tool_calls": [item.model_dump(mode="json") for item in plan.tool_calls], "decision_summary": plan.decision_summary},
                )
            return {"plan": plan.model_dump(mode="json")}
        needs_tools = (
            getattr(self.model, "requires_tool_descriptions", True)
            and not self.policy.is_task_planning_request(user_text)
        )
        tools = await self._tools() if needs_tools else []
        plan = await self.model.plan(user_text, tools, self._history(state))
        self.validate_plan(plan)
        if self.trace_recorder is not None:
            self.trace_recorder.record(
                state.get("trace_id", ""), "plan", "agent_plan", duration_ms=(time.perf_counter() - started) * 1000,
                data={
                    "intent": plan.intent.value,
                    "task_name": plan.task_name,
                    "dataset_name": plan.dataset_name,
                    "stage": plan.stage,
                    "tool_calls": [item.model_dump(mode="json") for item in plan.tool_calls],
                    "write_action": plan.write_action,
                    "decision_summary": plan.decision_summary,
                },
            )
        return {"plan": plan.model_dump(mode="json")}

    @staticmethod
    def _retrieval_query(user_text: str, plan: AgentPlan) -> str:
        anchors = {
            AgentIntent.PLATFORM_HEALTH: "Airflow health PostgreSQL metadata scheduler API",
            AgentIntent.TASK_STATUS: "task lifecycle DagRun queue priority pipeline",
            AgentIntent.TASK_DIAGNOSIS: "task stuck queue draining soft preemption checkpoint recovery",
            AgentIntent.GPU_DIAGNOSIS: "GPU 显存 reservation 独占 共享 资源等待 timeout",
            AgentIntent.STAGE_FAILURE: "stage failure validate log OOM checkpoint recovery",
            AgentIntent.LIST_TASKS: "task YAML pipeline stages task types",
            AgentIntent.GENERAL_READ: "platform architecture Airflow Docker GPU priority recovery",
            AgentIntent.PLATFORM_KNOWLEDGE: "平台规则",
            AgentIntent.TASK_PLANNING: "task YAML pipeline GPU concurrency images validation",
        }
        query = user_text.strip()
        anchor = anchors.get(plan.intent, "")
        if plan.stage:
            anchor = f"{anchor} stage {plan.stage}".strip()
        return f"{query}\n{anchor}".strip()

    async def retrieve_knowledge(self, state: AgentGraphState) -> dict[str, Any]:
        plan = self.validate_plan(AgentPlan.model_validate(state["plan"]))
        # Write requests use deterministic impact analysis from current MCP evidence;
        # retrieved text must never influence the frozen mutation arguments.
        if plan.intent in {AgentIntent.UNSUPPORTED_WRITE, AgentIntent.TASK_PLANNING} | WRITE_INTENTS or self.knowledge_retriever is None:
            if self.trace_recorder is not None:
                reason = "write_or_planning" if plan.intent in {AgentIntent.UNSUPPORTED_WRITE, AgentIntent.TASK_PLANNING} | WRITE_INTENTS else "retriever_unavailable"
                self.trace_recorder.record(state.get("trace_id", ""), "retrieval", "knowledge_retrieval", status="ok", data={"skipped": True, "reason": reason})
            return {"knowledge": []}
        query = self._retrieval_query(state.get("user_text", ""), plan)
        try:
            items = await self.knowledge_retriever.retrieve(query, top_k=self.knowledge_top_k)
        except Exception as exc:
            if self.trace_recorder is not None:
                self.trace_recorder.record(state.get("trace_id", ""), "retrieval", "knowledge_retrieval", status="error", data={"query": query, "error": str(exc)})
            return {
                "knowledge": [
                    KnowledgeObservation(
                        chunk_id="retrieval-error",
                        source_path="__retrieval__",
                        title="retrieval error",
                        section="",
                        content=str(exc),
                        score=0.0,
                        metadata={"error": True},
                    ).model_dump(mode="json")
                ]
            }
        result = []
        for item in items:
            if isinstance(item, KnowledgeObservation):
                obs = item
            else:
                data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
                obs = KnowledgeObservation.model_validate(data)
            result.append(obs.model_dump(mode="json"))
        if self.trace_recorder is not None:
            self.trace_recorder.record(
                state.get("trace_id", ""),
                "retrieval",
                "knowledge_retrieval",
                data={
                    "query": query,
                    "results": [
                        {"chunk_id": item.get("chunk_id"), "source": item.get("source_path"), "section": item.get("section"), "score": item.get("score")}
                        for item in result
                    ],
                },
            )
        return {"knowledge": result}

    async def execute_tools(self, state: AgentGraphState) -> dict[str, Any]:
        plan = self.validate_plan(AgentPlan.model_validate(state["plan"]))
        observations = await self.tool_client.execute(plan.tool_calls)
        return {"observations": [item.model_dump(mode="json") for item in observations]}

    @staticmethod
    def _trace(observations: list[ToolObservation]) -> list[dict[str, Any]]:
        return [
            {
                "tool": item.tool_name,
                "arguments": item.arguments,
                "ok": item.ok,
                "error": item.error,
            }
            for item in observations
        ]

    async def _task_planning_result(self, state: AgentGraphState, plan: AgentPlan):
        if self.task_planning_service is None:
            return None
        return self.task_planning_service.plan_from_draft(
            state.get("user_text", ""), plan.task_draft or {}
        )

    async def _write_answer(
        self,
        state: AgentGraphState,
        plan: AgentPlan,
        observations: list[ToolObservation],
    ) -> AgentResponse:
        if self.action_coordinator is None:
            return AgentResponse(
                intent=plan.intent,
                summary="Write action coordinator is unavailable; no mutation was executed.",
                confidence="low",
                blocked=True,
                errors=["write action coordinator is not configured"],
                tool_trace=self._trace(observations),
            )

        task_plan_dict = None
        if plan.intent == AgentIntent.SUBMIT_TASK:
            planning = await self._task_planning_result(state, plan)
            if planning is None:
                return AgentResponse(
                    intent=plan.intent,
                    summary="Task planning service is unavailable; submit cannot be prepared.",
                    confidence="low",
                    blocked=True,
                    errors=["task planning service is not configured"],
                    tool_trace=self._trace(observations),
                )
            task_plan_dict = planning.model_dump(mode="json")
            if self.trace_recorder is not None:
                self.trace_recorder.record(
                    state.get("trace_id", ""), "planning", "task_planning", status="ok" if planning.valid else "invalid",
                    data={"valid": planning.valid, "task_spec": planning.task_spec.model_dump(mode="json"), "issues": [item.model_dump(mode="json") for item in planning.issues]},
                )
            if not planning.valid:
                errors = [item.message for item in planning.errors]
                unresolved = ", ".join(planning.unresolved_fields) or "none"
                return AgentResponse(
                    intent=plan.intent,
                    summary=f"Submit was not prepared because TaskSpec is invalid: unresolved={unresolved}.",
                    evidence=["V0.6 deterministic TaskPlanningResult.valid=false."],
                    recommended_next_actions=["Correct the TaskSpec fields and request submit again."],
                    confidence="high",
                    blocked=True,
                    errors=errors,
                    task_plan=task_plan_dict,
                    tool_trace=self._trace(observations),
                )

        if plan.intent == AgentIntent.SET_TASK_PRIORITY:
            raw = (plan.write_action or {}).get("priority")
            if raw is None:
                return AgentResponse(
                    intent=plan.intent,
                    summary="Priority change was not prepared because the target numeric priority was not explicitly provided.",
                    evidence=["V0.8 does not infer a priority value for write operations."],
                    recommended_next_actions=["Specify an explicit priority, for example: 优先级改成5."],
                    confidence="high",
                    blocked=True,
                    tool_trace=self._trace(observations),
                )

        failed_reads = [item for item in observations if not item.ok]
        if failed_reads:
            return AgentResponse(
                intent=plan.intent,
                summary="Write action was not prepared because current-state impact evidence is incomplete.",
                evidence=[f"Read tool failed: {item.tool_name}" for item in failed_reads],
                recommended_next_actions=["Resolve the read-side platform error, then request the action again."],
                confidence="high",
                blocked=True,
                errors=[f"{item.tool_name}: {item.error}" for item in failed_reads],
                task_plan=task_plan_dict,
                tool_trace=self._trace(observations),
            )

        try:
            pending = await self.action_coordinator.prepare(
                state_user_text=state.get("user_text", ""),
                thread_id=state.get("thread_id", "default"),
                plan=plan,
                observations=observations,
                task_plan=task_plan_dict,
                trace_id=state.get("trace_id", ""),
            )
        except Exception as exc:
            return AgentResponse(
                intent=plan.intent,
                summary="Write action could not be prepared; no mutation was executed.",
                confidence="high",
                blocked=True,
                errors=[str(exc)],
                task_plan=task_plan_dict,
                tool_trace=self._trace(observations),
            )

        pending_payload = pending.model_dump(mode="json")
        return AgentResponse(
            intent=plan.intent,
            summary=(
                f"Approval required before {pending.tool_name}. "
                f"risk={pending.risk_level}; approval_id={pending.approval_id}. No mutation has run yet."
            ),
            evidence=list(pending.impact_details),
            recommended_next_actions=[
                f"Approve with: dataops-agent approve {pending.approval_id}",
                f"Reject with: dataops-agent reject {pending.approval_id}",
            ],
            confidence="high",
            blocked=True,
            approval_required=True,
            approval_id=pending.approval_id,
            pending_action=pending_payload,
            task_plan=task_plan_dict,
            tool_trace=self._trace(observations),
        )

    async def answer(self, state: AgentGraphState) -> dict[str, Any]:
        plan = AgentPlan.model_validate(state["plan"])
        observations = [ToolObservation.model_validate(item) for item in state.get("observations", [])]
        raw_knowledge = [KnowledgeObservation.model_validate(item) for item in state.get("knowledge", [])]
        retrieval_errors = [
            item for item in raw_knowledge if item.metadata.get("error") or item.source_path == "__retrieval__"
        ]
        knowledge = [item for item in raw_knowledge if item not in retrieval_errors]

        if plan.intent in WRITE_INTENTS:
            response = await self._write_answer(state, plan, observations)
            return {"response": response.model_dump(mode="json")}

        if plan.intent == AgentIntent.TASK_PLANNING:
            planning = await self._task_planning_result(state, plan)
            if planning is None:
                response = AgentResponse(
                    intent=plan.intent,
                    summary="Task planning service is unavailable.",
                    confidence="low",
                    blocked=True,
                    errors=["task planning service is not configured"],
                )
                return {"response": response.model_dump(mode="json")}
            if self.trace_recorder is not None:
                self.trace_recorder.record(
                    state.get("trace_id", ""), "planning", "task_planning", status="ok" if planning.valid else "invalid",
                    data={"valid": planning.valid, "task_spec": planning.task_spec.model_dump(mode="json"), "issues": [item.model_dump(mode="json") for item in planning.issues]},
                )
            errors = [item.message for item in planning.errors]
            warnings = [item.message for item in planning.warnings]
            if planning.valid:
                summary = (
                    f"TaskSpec is valid. prefix={planning.task_spec.task_prefix}; "
                    f"priority={planning.resolved_priority} ({planning.priority_source}); "
                    f"datasets={len(planning.task_spec.datasets)}; no task was submitted."
                )
                confidence = "high"
            else:
                unresolved = ", ".join(planning.unresolved_fields) or "none"
                summary = f"TaskSpec is not ready: unresolved={unresolved}; no task was submitted."
                confidence = "high"
            response = AgentResponse(
                intent=plan.intent,
                summary=summary,
                evidence=[
                    f"Deterministic platform validation: {'passed' if planning.valid else 'failed'}.",
                    f"Defaults used: {', '.join(planning.defaults_used) or 'none'}.",
                ],
                recommended_next_actions=(
                    [] if planning.valid else ["Provide unresolved fields or correct validation errors, then plan again."]
                ),
                confidence=confidence,
                blocked=False,
                errors=errors + warnings,
                task_plan=planning.model_dump(mode="json"),
            )
            return {"response": response.model_dump(mode="json")}

        response = await self.model.synthesize(
            state.get("user_text", ""),
            plan,
            observations,
            self._history(state),
            knowledge,
        )
        response.intent = plan.intent
        if plan.intent == AgentIntent.UNSUPPORTED_WRITE:
            response.blocked = True
        for item in retrieval_errors:
            response.errors.append(f"knowledge retrieval: {item.content}")
        return {"response": response.model_dump(mode="json")}

    @staticmethod
    def route_after_plan(state: AgentGraphState) -> str:
        plan = AgentPlan.model_validate(state["plan"])
        return "tools" if plan.tool_calls else "answer"

    @staticmethod
    def route_after_retrieval(state: AgentGraphState) -> str:
        plan = AgentPlan.model_validate(state["plan"])
        return "tools" if plan.tool_calls else "answer"


class BaseReadOnlyAgent:
    def __init__(
        self,
        nodes: ReadOnlyAgentNodes,
        conversation_store: ConversationStore,
        action_coordinator: WriteActionCoordinator | None = None,
        trace_recorder=None,
    ):
        self.nodes = nodes
        self.conversation_store = conversation_store
        self.action_coordinator = action_coordinator
        self.trace_recorder = trace_recorder

    def _initial_state(self, user_text: str, thread_id: str, trace_id: str = "") -> AgentGraphState:
        history = self.conversation_store.load(thread_id)
        return {
            "user_text": user_text,
            "thread_id": thread_id,
            "history": [turn.model_dump(mode="json") for turn in history],
            "knowledge": [],
            "observations": [],
            "trace_id": trace_id,
        }

    def _commit(self, thread_id: str, user_text: str, response: AgentResponse) -> None:
        self.conversation_store.append(
            thread_id,
            ConversationTurn(
                user=user_text,
                assistant_summary=response.summary,
                intent=response.intent,
            ),
        )

    async def approve(self, approval_id: str):
        if self.action_coordinator is None:
            raise RuntimeError("Write action coordinator is not configured")
        preview = self.action_coordinator.approval_store.get(approval_id)
        if self.trace_recorder is None:
            return await self.action_coordinator.execute_approval(approval_id)
        trace_id = self.trace_recorder.start_trace(
            kind="approval_execution", user_request=f"approve {approval_id}", thread_id=preview.thread_id, parent_trace_id=preview.trace_id or None
        )
        try:
            with self.trace_recorder.activate(trace_id):
                item = await self.action_coordinator.execute_approval(approval_id, execution_trace_id=trace_id)
            self.trace_recorder.finish(trace_id, status=item.status, intent=preview.tool_name, response_summary=f"approval {approval_id}: {item.status}", errors=[item.error] if item.error else [])
            return item
        except Exception as exc:
            self.trace_recorder.record(trace_id, "error", "approval_execution", status="error", data={"error": str(exc), "approval_id": approval_id})
            self.trace_recorder.finish(trace_id, status="error", intent=preview.tool_name, errors=[str(exc)])
            raise

    def reject(self, approval_id: str, reason: str = "Rejected by user"):
        if self.action_coordinator is None:
            raise RuntimeError("Write action coordinator is not configured")
        preview = self.action_coordinator.approval_store.get(approval_id)
        if self.trace_recorder is None:
            return self.action_coordinator.approval_store.reject(approval_id, reason)
        trace_id = self.trace_recorder.start_trace(kind="approval_reject", user_request=f"reject {approval_id}", thread_id=preview.thread_id, parent_trace_id=preview.trace_id or None)
        try:
            with self.trace_recorder.activate(trace_id):
                item = self.action_coordinator.approval_store.reject(approval_id, reason)
                self.trace_recorder.record(trace_id, "approval", "approval_rejected", status="ok", data={"approval_id": approval_id, "reason": reason, "origin_trace_id": preview.trace_id})
            self.trace_recorder.finish(trace_id, status="rejected", intent=preview.tool_name, response_summary=f"approval {approval_id}: rejected")
            return item
        except Exception as exc:
            self.trace_recorder.finish(trace_id, status="error", intent=preview.tool_name, errors=[str(exc)])
            raise

    def approvals(self, status: str = "pending"):
        if self.action_coordinator is None:
            return []
        return self.action_coordinator.approval_store.list(status=status)


class SequentialReadOnlyAgent(BaseReadOnlyAgent):
    """Dependency-light runtime used for deterministic local tests."""

    async def run(self, user_text: str, thread_id: str = "default") -> AgentResponse:
        trace_id = self.trace_recorder.start_trace(kind="agent_request", user_request=user_text, thread_id=thread_id) if self.trace_recorder is not None else ""
        state = self._initial_state(user_text, thread_id, trace_id)
        try:
            context = self.trace_recorder.activate(trace_id) if self.trace_recorder is not None else None
            if context is None:
                state.update(await self.nodes.plan(state))
                state.update(await self.nodes.retrieve_knowledge(state))
                if self.nodes.route_after_retrieval(state) == "tools":
                    state.update(await self.nodes.execute_tools(state))
                state.update(await self.nodes.answer(state))
            else:
                with context:
                    state.update(await self.nodes.plan(state))
                    state.update(await self.nodes.retrieve_knowledge(state))
                    if self.nodes.route_after_retrieval(state) == "tools":
                        state.update(await self.nodes.execute_tools(state))
                    state.update(await self.nodes.answer(state))
            response = AgentResponse.model_validate(state["response"])
            response.trace_id = trace_id or None
            self._commit(thread_id, user_text, response)
            if self.trace_recorder is not None:
                self.trace_recorder.finish(trace_id, status="ok", intent=response.intent.value, response_summary=response.summary, errors=response.errors)
            return response
        except Exception as exc:
            if self.trace_recorder is not None:
                self.trace_recorder.record(trace_id, "error", "agent_request", status="error", data={"error": str(exc)})
                self.trace_recorder.finish(trace_id, status="error", errors=[str(exc)])
            raise


class LangGraphReadOnlyAgent(BaseReadOnlyAgent):
    """Stateful runtime backed by LangGraph; write execution stays outside model nodes."""

    def __init__(self, nodes: ReadOnlyAgentNodes, conversation_store: ConversationStore, action_coordinator=None, trace_recorder=None):
        super().__init__(nodes, conversation_store, action_coordinator, trace_recorder=trace_recorder)
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "LangGraph is not installed. Install requirements-agent.txt or set PLATFORM_AGENT_RUNTIME=sequential for dependency-light tests."
            ) from exc

        builder = StateGraph(AgentGraphState)
        builder.add_node("plan", self.nodes.plan)
        builder.add_node("retrieve", self.nodes.retrieve_knowledge)
        builder.add_node("tools", self.nodes.execute_tools)
        builder.add_node("answer", self.nodes.answer)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan", self.nodes.route_after_plan, {"tools": "retrieve", "answer": "retrieve"}
        )
        builder.add_conditional_edges(
            "retrieve", self.nodes.route_after_retrieval, {"tools": "tools", "answer": "answer"}
        )
        builder.add_edge("tools", "answer")
        builder.add_edge("answer", END)
        return builder.compile(checkpointer=InMemorySaver())

    async def run(self, user_text: str, thread_id: str = "default") -> AgentResponse:
        trace_id = self.trace_recorder.start_trace(kind="agent_request", user_request=user_text, thread_id=thread_id) if self.trace_recorder is not None else ""
        initial = self._initial_state(user_text, thread_id, trace_id)
        try:
            if self.trace_recorder is None:
                result = await self.graph.ainvoke(initial, config={"configurable": {"thread_id": thread_id}})
            else:
                with self.trace_recorder.activate(trace_id):
                    result = await self.graph.ainvoke(initial, config={"configurable": {"thread_id": thread_id}})
            response = AgentResponse.model_validate(result["response"])
            response.trace_id = trace_id or None
            self._commit(thread_id, user_text, response)
            if self.trace_recorder is not None:
                self.trace_recorder.finish(trace_id, status="ok", intent=response.intent.value, response_summary=response.summary, errors=response.errors)
            return response
        except Exception as exc:
            if self.trace_recorder is not None:
                self.trace_recorder.record(trace_id, "error", "agent_request", status="error", data={"error": str(exc)})
                self.trace_recorder.finish(trace_id, status="error", errors=[str(exc)])
            raise


def build_agent_runtime(
    runtime: str,
    model,
    tool_client,
    conversation_store: ConversationStore,
    max_tool_calls: int = 6,
    knowledge_retriever=None,
    knowledge_top_k: int = 5,
    task_planning_service=None,
    approval_store=None,
    action_verifier=None,
    trace_recorder=None,
):
    policy = AgentPolicyEngine(max_tool_calls=max_tool_calls)
    action_coordinator = (
        WriteActionCoordinator(tool_client, policy, approval_store, verifier=action_verifier, trace_recorder=trace_recorder)
        if approval_store is not None
        else None
    )
    nodes = ReadOnlyAgentNodes(
        model=model,
        tool_client=tool_client,
        policy=policy,
        knowledge_retriever=knowledge_retriever,
        knowledge_top_k=knowledge_top_k,
        task_planning_service=task_planning_service,
        action_coordinator=action_coordinator,
        trace_recorder=trace_recorder,
    )
    runtime = (runtime or "langgraph").strip().lower()
    if runtime in {"sequential", "test"}:
        return SequentialReadOnlyAgent(nodes, conversation_store, action_coordinator, trace_recorder=trace_recorder)
    if runtime == "langgraph":
        return LangGraphReadOnlyAgent(nodes, conversation_store, action_coordinator, trace_recorder=trace_recorder)
    raise ValueError(f"Unsupported PLATFORM_AGENT_RUNTIME: {runtime}")
