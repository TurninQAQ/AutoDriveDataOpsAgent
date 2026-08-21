from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from platform_mcp.server import READ_ONLY_TOOL_NAMES

from .actions import WRITE_INTENT_TO_TOOL, WriteActionCoordinator
from .adaptive import AdaptiveLimits, AdaptiveLoopController, READ_ONLY_INTENTS
from .autonomy import AutonomyMode, BoundedAutonomyPolicy
from .memory import ConversationStore
from .models import (
    AgentIntent,
    AgentPlan,
    AgentResponse,
    ConversationTurn,
    GoalContract,
    GoalEvaluation,
    KnowledgeObservation,
    GoalProgress,
    ToolObservation,
)
from .goal import (
    evaluate_goal_progress,
    finalize_goal_response,
    normalize_plan_goal,
    resolve_goal_contract,
)
from .policy import AgentPolicyEngine


WRITE_INTENTS = frozenset(WRITE_INTENT_TO_TOOL)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_search_knowledge(observation: ToolObservation) -> list[KnowledgeObservation]:
    """Normalize a successful MCP knowledge result into shared evidence."""
    if observation.tool_name != "search_knowledge" or not observation.ok:
        return []
    payload = observation.data if isinstance(observation.data, dict) else {}
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []

    normalized: list[KnowledgeObservation] = []
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or raw.get("source_path") or "").strip()
        source_path = str(raw.get("source_path") or "").strip()
        section = str(raw.get("section") or "").strip()
        if not source_path and source:
            source_path, separator, source_section = source.partition("#")
            if separator and not section:
                section = source_section
        source_path = source_path.strip()
        content = str(raw.get("content") or "")
        if not source_path and not content:
            continue

        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
        if raw.get("rank") is not None:
            metadata.setdefault("rank", raw.get("rank"))
        chunk_id = str(raw.get("chunk_id") or "").strip()
        if not chunk_id:
            chunk_id = f"{source_path}#{section}" if section else f"{source_path}:{index}"
        normalized.append(
            KnowledgeObservation(
                chunk_id=chunk_id,
                source_path=source_path or "__knowledge__",
                title=str(raw.get("title") or source_path or "platform knowledge"),
                section=section,
                content=content,
                score=_as_float(raw.get("score")),
                lexical_score=_as_float(raw.get("lexical_score")),
                vector_score=_as_float(raw.get("vector_score")),
                metadata=metadata,
            )
        )
    return normalized


def merge_knowledge_observations(
    *groups: list[KnowledgeObservation],
) -> list[KnowledgeObservation]:
    """Merge legacy and MCP knowledge evidence without duplicate chunks."""
    merged: list[KnowledgeObservation] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = item.chunk_id or f"{item.citation}\n{item.content}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


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
    request_id: str
    adaptive_steps: list[dict[str, Any]]
    adaptive_step_count: int
    tool_call_count: int
    current_intent: str
    evidence_sufficient: bool
    termination_reason: str
    adaptive_errors: list[str]
    evidence_records: list[dict[str, Any]]
    goal: dict[str, Any]
    goal_contract: dict[str, Any]
    goal_evaluation: dict[str, Any]
    goal_progress: str
    goal_explicit: bool


class ReadOnlyAgentNodes:
    """Agent nodes shared by sequential and LangGraph runtimes.

    The historical class name is retained for compatibility. In V0.8 normal model
    tool calls are still read-only; state-changing calls can only be executed by
    the separate guarded WriteActionCoordinator after persisted HITL approval or
    the deterministic V1.7 resume policy.
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
        max_steps: int = 8,
        max_identical_tool_calls: int = 2,
        max_consecutive_tool_failures: int = 2,
        autonomy_policy: BoundedAutonomyPolicy | None = None,
    ):
        self.model = model
        self.tool_client = tool_client
        self.policy = policy
        self.knowledge_retriever = knowledge_retriever
        self.knowledge_top_k = max(1, knowledge_top_k)
        self.task_planning_service = task_planning_service
        self.action_coordinator = action_coordinator
        self.autonomy_policy = autonomy_policy
        self.trace_recorder = trace_recorder
        self.adaptive_limits = AdaptiveLimits(
            max_steps=max_steps,
            max_tool_calls=policy.max_tool_calls,
            max_identical_tool_calls=max_identical_tool_calls,
            max_consecutive_tool_failures=max_consecutive_tool_failures,
        )
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
            plan = normalize_plan_goal(plan)
            goal_contract = resolve_goal_contract(plan.goal.goal_type, plan.intent)
            self.validate_plan(plan)
            if self.trace_recorder is not None:
                self.trace_recorder.record(
                    state.get("trace_id", ""), "plan", "agent_plan", duration_ms=(time.perf_counter() - started) * 1000,
                    data={"intent": plan.intent.value, "goal": plan.goal.model_dump(mode="json") if plan.goal else None, "tool_calls": [item.model_dump(mode="json") for item in plan.tool_calls], "decision_summary": plan.decision_summary},
                )
            return {
                "plan": plan.model_dump(mode="json"),
                "goal_contract": goal_contract.model_dump(mode="json"),
            }
        needs_tools = (
            getattr(self.model, "requires_tool_descriptions", True)
            and not self.policy.is_task_planning_request(user_text)
        )
        tools = await self._tools() if needs_tools else []
        plan = await self.model.plan(user_text, tools, self._history(state))
        goal_explicit = plan.goal is not None
        plan = normalize_plan_goal(plan)
        goal_contract = resolve_goal_contract(plan.goal.goal_type, plan.intent)
        self.validate_plan(plan)
        if self.trace_recorder is not None:
            self.trace_recorder.record(
                state.get("trace_id", ""), "plan", "agent_plan", duration_ms=(time.perf_counter() - started) * 1000,
                data={
                    "intent": plan.intent.value,
                    "goal": plan.goal.model_dump(mode="json") if plan.goal else None,
                    "goal_contract": goal_contract.model_dump(mode="json"),
                    "task_name": plan.task_name,
                    "dataset_name": plan.dataset_name,
                    "stage": plan.stage,
                    "tool_calls": [item.model_dump(mode="json") for item in plan.tool_calls],
                    "write_action": plan.write_action,
                    "decision_summary": plan.decision_summary,
                },
            )
        return {
            "plan": plan.model_dump(mode="json"),
            "goal_contract": goal_contract.model_dump(mode="json"),
            "goal_explicit": goal_explicit,
        }

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

    def adaptive_supported(self) -> bool:
        return callable(getattr(self.model, "decide_next", None))

    async def adaptive_read(self, state: AgentGraphState) -> dict[str, Any]:
        """Run the shared one-read-tool-per-step controller."""

        plan = self.validate_plan(AgentPlan.model_validate(state["plan"]))
        observations = [ToolObservation.model_validate(item) for item in state.get("observations", [])]
        knowledge = [KnowledgeObservation.model_validate(item) for item in state.get("knowledge", [])]
        evidence_records = list(state.get("evidence_records", []))
        trace_id = state.get("trace_id", "")

        def trace_event(name: str, *, status: str = "ok", data: dict[str, Any] | None = None) -> None:
            if self.trace_recorder is not None:
                self.trace_recorder.record(trace_id, "adaptive", name, status=status, data=data or {})

        async def execute_one(call):
            self.policy.validate_tool_name(call.name)
            return await self.tool_client.execute([call])

        controller = AdaptiveLoopController(
            self.model,
            self.policy,
            self.adaptive_limits,
            trace_event=trace_event,
        )
        result = await controller.run(
            user_text=state.get("user_text", ""),
            initial_plan=plan,
            tool_descriptions=await self._tools(),
            observations=observations,
            knowledge=knowledge,
            history=self._history(state),
            execute_tool=execute_one,
            normalize_observation=normalize_search_knowledge,
            initial_intent=plan.intent,
            evidence_records=evidence_records,
            goal=plan.goal,
            goal_contract=state.get("goal_contract"),
            goal_aware=bool(state.get("goal_explicit", False)),
        )
        return {
            "observations": [item.model_dump(mode="json") for item in result.observations],
            "knowledge": [item.model_dump(mode="json") for item in result.knowledge],
            "adaptive_steps": result.steps,
            "adaptive_step_count": len(result.steps),
            "tool_call_count": result.tool_call_count,
            "current_intent": result.current_intent.value if result.current_intent else plan.intent.value,
            "evidence_sufficient": result.evidence_sufficient,
            "termination_reason": result.termination_reason,
            "adaptive_errors": result.errors,
            "evidence_records": [item.as_dict() for item in result.evidence_records],
            "goal": result.goal.model_dump(mode="json") if result.goal else None,
            "goal_contract": result.goal_contract.model_dump(mode="json") if result.goal_contract else None,
            "goal_evaluation": result.goal_evaluation.model_dump(mode="json") if result.goal_evaluation else None,
            "goal_progress": result.goal_evaluation.state.value if result.goal_evaluation else None,
        }

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

    @staticmethod
    def _attach_response_metadata(
        response: AgentResponse,
        plan: AgentPlan,
        state: AgentGraphState,
    ) -> AgentResponse:
        plan = normalize_plan_goal(plan)
        response.initial_plan = plan.model_dump(mode="json")
        response.goal = plan.goal
        if state.get("goal_progress"):
            try:
                response.goal_progress = GoalProgress(state["goal_progress"])
                response.goal = plan.goal.model_copy(
                    update={"completion_state": response.goal_progress}
                )
            except ValueError:
                response.goal_progress = None
        if state.get("adaptive_step_count", 0):
            response.termination_reason = state.get("termination_reason")
            response.adaptive_step_count = int(state.get("adaptive_step_count", 0))
            response.evidence_sufficient = bool(state.get("evidence_sufficient", False))
            response.adaptive_steps = list(state.get("adaptive_steps", []))
            for error in list(state.get("adaptive_errors", [])):
                if error not in response.errors:
                    response.errors.append(error)
            if not response.evidence_sufficient:
                reason = response.termination_reason or "adaptive evidence incomplete"
                message = f"Adaptive evidence was not sufficient ({reason})."
                if message not in response.errors:
                    response.errors.append(message)
                goal_state = state.get("goal_progress")
                if goal_state in {GoalProgress.IN_PROGRESS.value, GoalProgress.BLOCKED.value}:
                    goal_message = f"User goal was not fully verified ({goal_state})."
                    if goal_message not in response.errors:
                        response.errors.append(goal_message)
                response.confidence = "low"
                if response.goal_progress == GoalProgress.BLOCKED:
                    response.blocked = True
        return response

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
            request_id = state.get("trace_id") or state.get("request_id") or state.get("thread_id", "default")
            decision, pending = await self.action_coordinator.prepare_with_autonomy(
                state_user_text=state.get("user_text", ""),
                thread_id=state.get("thread_id", "default"),
                plan=plan,
                observations=observations,
                task_plan=task_plan_dict,
                trace_id=request_id,
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

        decision_payload = decision.model_dump(mode="json")
        if self.trace_recorder is not None and state.get("trace_id", ""):
            self.trace_recorder.record(
                state.get("trace_id", ""),
                "autonomy",
                "autonomy_policy",
                status=decision.mode.value.lower(),
                data=decision_payload,
            )
            self.trace_recorder.record(
                state.get("trace_id", ""),
                "autonomy",
                "autonomy_budget",
                status="ok",
                data=decision.budget,
            )

        if decision.mode == AutonomyMode.DENY:
            return AgentResponse(
                intent=plan.intent,
                summary=f"{decision.action} was denied by the deterministic autonomy policy; no mutation was executed.",
                evidence=list(decision.reasons),
                recommended_next_actions=["Review the current target/state evidence and request a guarded action again."],
                confidence="high",
                blocked=True,
                errors=list(decision.reasons),
                policy_decision=decision_payload,
                authorization_mode="hitl",
                task_plan=task_plan_dict,
                tool_trace=self._trace(observations),
            )

        if decision.mode == AutonomyMode.AUTO and decision.reservation_status == "duplicate_existing":
            existing = pending
            if existing is None:
                return AgentResponse(
                    intent=plan.intent,
                    summary="Duplicate AUTO request was blocked because its existing authorization record could not be loaded.",
                    confidence="low",
                    blocked=True,
                    errors=["duplicate_existing authorization record missing"],
                    policy_decision=decision_payload,
                    authorization_mode="auto",
                    task_plan=task_plan_dict,
                    tool_trace=self._trace(observations),
                )
            existing_goal = existing.goal_verification_result or {}
            already_satisfied = existing.status == "executed" and existing_goal.get("status") == "satisfied"
            return AgentResponse(
                intent=plan.intent,
                summary=(
                    "The same AUTO action already has an authorization/execution record; "
                    "no duplicate mutation was executed."
                ),
                evidence=["duplicate_existing"],
                confidence="high" if already_satisfied else "medium",
                blocked=not already_satisfied,
                approval_required=False,
                approval_id=existing.approval_id,
                pending_action=existing.model_dump(mode="json"),
                action_result=existing.execution_result,
                authorization_mode="auto",
                policy_decision=decision_payload,
                action_verification=existing.verification_result,
                goal_verification_result=existing.goal_verification_result,
                task_plan=task_plan_dict,
                tool_trace=self._trace(observations),
            )

        if decision.mode == AutonomyMode.AUTO:
            if pending is None:
                return AgentResponse(
                    intent=plan.intent,
                    summary="Autonomy policy allowed resume_task but no execution record was created; no mutation was executed.",
                    confidence="low",
                    blocked=True,
                    errors=["auto execution record missing"],
                    policy_decision=decision_payload,
                    authorization_mode="auto",
                    task_plan=task_plan_dict,
                    tool_trace=self._trace(observations),
                )
            if self.trace_recorder is not None and state.get("trace_id", ""):
                self.trace_recorder.record(
                    state.get("trace_id", ""), "autonomy", "autonomous_execution", status="started",
                    data={"approval_id": pending.approval_id, "frozen_arguments": pending.arguments, "automatic_retry": False},
                )
            try:
                executed = await self.action_coordinator.execute_approval(
                    pending.approval_id,
                    execution_trace_id=state.get("trace_id", ""),
                )
            except Exception as exc:
                return AgentResponse(
                    intent=plan.intent,
                    summary="Autonomous resume execution could not be completed; operator review is required.",
                    confidence="low",
                    blocked=True,
                    errors=[str(exc)],
                    approval_id=pending.approval_id,
                    pending_action=pending.model_dump(mode="json"),
                    authorization_mode="auto",
                    policy_decision=decision_payload,
                    task_plan=task_plan_dict,
                    tool_trace=self._trace(observations),
                )

            goal_payload = executed.goal_verification_result or {}
            goal_status = str(goal_payload.get("status") or "")
            if executed.status == "executed" and goal_status == "satisfied":
                summary = "Autonomous resume completed and the user goal was deterministically verified."
                termination_reason = "goal_satisfied"
                blocked = False
                confidence = "high"
            elif executed.status == "executed" and goal_status == "in_progress":
                summary = "Autonomous resume executed, but post-action evidence is still incomplete; no automatic retry was performed."
                termination_reason = "goal_incomplete"
                blocked = True
                confidence = "medium"
            elif executed.status == "executed" and goal_status in {"failed", "inconclusive"}:
                summary = f"Autonomous resume executed, but Goal Verification is {goal_status}; operator review is required."
                termination_reason = "goal_blocked"
                blocked = True
                confidence = "low"
            else:
                summary = "Autonomous resume did not pass the guarded execution chain; operator review is required."
                termination_reason = "action_verification_failed"
                blocked = True
                confidence = "low"
            return AgentResponse(
                intent=plan.intent,
                summary=summary,
                evidence=list(pending.impact_details),
                recommended_next_actions=([] if not blocked else ["Review the deterministic verification and escalation record; do not retry automatically."]),
                confidence=confidence,
                blocked=blocked,
                approval_required=False,
                approval_id=executed.approval_id,
                pending_action=executed.model_dump(mode="json"),
                action_result=executed.execution_result,
                termination_reason=termination_reason,
                authorization_mode="auto",
                policy_decision=decision_payload,
                action_verification=executed.verification_result,
                goal_verification_result=goal_payload,
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
            authorization_mode="hitl",
            policy_decision=decision_payload,
            task_plan=task_plan_dict,
            tool_trace=self._trace(observations),
        )

    async def answer(self, state: AgentGraphState) -> dict[str, Any]:
        plan = normalize_plan_goal(AgentPlan.model_validate(state["plan"]))
        observations = [ToolObservation.model_validate(item) for item in state.get("observations", [])]
        raw_knowledge = [KnowledgeObservation.model_validate(item) for item in state.get("knowledge", [])]
        retrieval_errors = [
            item for item in raw_knowledge if item.metadata.get("error") or item.source_path == "__retrieval__"
        ]
        legacy_knowledge = [item for item in raw_knowledge if item not in retrieval_errors]
        tool_knowledge = [
            item
            for observation in observations
            for item in normalize_search_knowledge(observation)
        ]
        knowledge = merge_knowledge_observations(legacy_knowledge, tool_knowledge)
        goal_contract = (
            GoalContract.model_validate(state["goal_contract"])
            if state.get("goal_contract")
            else resolve_goal_contract(plan.goal.goal_type, plan.intent)
        )
        goal_evaluation = evaluate_goal_progress(
            plan.goal,
            state.get("evidence_records", []),
            observations,
            knowledge,
            goal_contract=goal_contract,
        )
        # Adaptive budget/failure termination is an authoritative blocked state;
        # do not let the answer node recompute it back to IN_PROGRESS merely
        # because the final observation set is incomplete.
        if state.get("goal_progress") == GoalProgress.BLOCKED.value:
            goal_evaluation = goal_evaluation.model_copy(
                update={
                    "state": GoalProgress.BLOCKED,
                    "summary": "Goal was blocked by adaptive termination before completion.",
                }
            )
        if state.get("termination_reason") == "unsafe_adaptive_decision":
            goal_evaluation = goal_evaluation.model_copy(
                update={
                    "state": GoalProgress.BLOCKED,
                    "summary": "Goal was blocked because an adaptive decision crossed the read-only safety boundary.",
                }
            )
        state["goal_evaluation"] = goal_evaluation.model_dump(mode="json")
        state["goal_progress"] = goal_evaluation.state.value

        if plan.intent in WRITE_INTENTS:
            response = await self._write_answer(state, plan, observations)
            if response.authorization_mode == "auto":
                goal_status = str((response.goal_verification_result or {}).get("status") or "")
                if goal_status == "satisfied":
                    write_state = GoalProgress.SATISFIED
                    satisfied = ["RESUME_GOAL_VERIFIED"]
                    missing = []
                    summary = "Resume action and user goal were deterministically verified."
                elif goal_status == "in_progress":
                    write_state = GoalProgress.IN_PROGRESS
                    satisfied = ["RESUME_ACTION_EXECUTED"]
                    missing = ["RESUME_GOAL_VERIFIED"]
                    summary = "Resume action executed, but the user goal remains in progress."
                else:
                    write_state = GoalProgress.BLOCKED
                    satisfied = ["RESUME_ACTION_EXECUTED"] if response.action_result else []
                    missing = ["RESUME_GOAL_VERIFIED"]
                    summary = "Autonomous resume did not complete the user goal."
                state["goal_progress"] = write_state.value
                state["goal_evaluation"] = GoalEvaluation(
                    state=write_state,
                    satisfied_conditions=satisfied,
                    missing_conditions=missing,
                    summary=summary,
                ).model_dump(mode="json")
            else:
                state["goal_progress"] = (
                    GoalProgress.SATISFIED.value
                    if response.approval_required
                    else GoalProgress.BLOCKED.value
                )
                state["goal_evaluation"] = GoalEvaluation(
                    state=GoalProgress.SATISFIED if response.approval_required else GoalProgress.BLOCKED,
                    satisfied_conditions=(
                        ["WRITE_PLAN_PREPARED"] if response.approval_required else []
                    ),
                    missing_conditions=(
                        [] if response.approval_required else ["WRITE_PLAN_PREPARED"]
                    ),
                    summary=(
                        "Write plan is prepared and awaiting HITL approval."
                        if response.approval_required
                        else "Write plan could not be prepared."
                    ),
                ).model_dump(mode="json")
            return {"response": self._attach_response_metadata(response, plan, state).model_dump(mode="json")}

        if plan.intent == AgentIntent.TASK_PLANNING:
            planning = await self._task_planning_result(state, plan)
            if planning is None:
                state["goal_progress"] = GoalProgress.BLOCKED.value
                state["goal_evaluation"] = GoalEvaluation(
                    state=GoalProgress.BLOCKED,
                    missing_conditions=["TASK_PLAN_VALIDATED"],
                    summary="Task planning service is unavailable.",
                ).model_dump(mode="json")
                response = AgentResponse(
                    intent=plan.intent,
                    summary="Task planning service is unavailable.",
                    confidence="low",
                    blocked=True,
                    errors=["task planning service is not configured"],
                )
                return {"response": self._attach_response_metadata(response, plan, state).model_dump(mode="json")}
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
            state["goal_progress"] = (
                GoalProgress.SATISFIED.value if planning.valid else GoalProgress.IN_PROGRESS.value
            )
            state["goal_evaluation"] = GoalEvaluation(
                state=GoalProgress.SATISFIED if planning.valid else GoalProgress.IN_PROGRESS,
                satisfied_conditions=["TASK_PLAN_VALIDATED"] if planning.valid else [],
                missing_conditions=[] if planning.valid else ["TASK_PLAN_VALIDATED"],
                summary=(
                    "Task plan passed deterministic validation."
                    if planning.valid
                    else "Task plan remains incomplete or invalid."
                ),
            ).model_dump(mode="json")
            return {"response": self._attach_response_metadata(response, plan, state).model_dump(mode="json")}

        final_intent = plan.intent
        if plan.intent in READ_ONLY_INTENTS and state.get("current_intent"):
            try:
                candidate = AgentIntent(state["current_intent"])
                if candidate in READ_ONLY_INTENTS:
                    final_intent = candidate
            except ValueError:
                pass
        synthesis_plan = plan
        if final_intent != plan.intent or state.get("adaptive_step_count", 0):
            synthesis_plan = plan.model_copy(update={"intent": final_intent, "tool_calls": []})

        response = await self.model.synthesize(
            state.get("user_text", ""),
            synthesis_plan,
            observations,
            self._history(state),
            knowledge,
        )
        response.intent = final_intent
        response.tool_trace = self._trace(observations)
        response.knowledge_sources = list(dict.fromkeys(item.citation for item in knowledge))
        response.retrieval_trace = [
            {
                "chunk_id": item.chunk_id,
                "source": item.citation,
                "score": item.score,
                **(
                    {"rank": item.metadata["rank"]}
                    if item.metadata.get("rank") is not None
                    else {}
                ),
            }
            for item in knowledge
        ]
        if plan.intent == AgentIntent.UNSUPPORTED_WRITE:
            response.blocked = True
        for item in retrieval_errors:
            response.errors.append(f"knowledge retrieval: {item.content}")
        final_goal_evaluation = finalize_goal_response(
            plan.goal,
            goal_contract,
            goal_evaluation,
            response,
        )
        state["goal_evaluation"] = final_goal_evaluation.model_dump(mode="json")
        state["goal_progress"] = final_goal_evaluation.state.value
        if final_goal_evaluation.state != GoalProgress.SATISFIED:
            state["evidence_sufficient"] = False
            if state.get("adaptive_step_count") and state.get("termination_reason") in {
                None,
                "unknown",
                "agent_finished",
                "goal_satisfied",
                "goal_incomplete",
            }:
                state["termination_reason"] = "goal_incomplete"
            message = "Final synthesis did not provide enough information to complete the user goal."
            if message not in response.errors:
                response.errors.append(message)
            if response.confidence == "high":
                response.confidence = "low"
        else:
            state["evidence_sufficient"] = True
            if state.get("adaptive_step_count") and state.get("termination_reason") == "goal_incomplete":
                state["termination_reason"] = "goal_satisfied"
        return {"response": self._attach_response_metadata(response, plan, state).model_dump(mode="json")}

    @staticmethod
    def route_after_plan(state: AgentGraphState) -> str:
        # Keep the historical router result for lightweight fake LangGraph
        # implementations. The real graph maps both branches to the compatibility
        # retrieval node before applying the adaptive/legacy route.
        plan = AgentPlan.model_validate(state["plan"])
        return "tools" if plan.tool_calls else "answer"

    def route_after_retrieval(self, state: AgentGraphState) -> str:
        plan = AgentPlan.model_validate(state["plan"])
        if plan.intent in READ_ONLY_INTENTS and self.adaptive_supported():
            return "adaptive"
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
            "adaptive_steps": [],
            "adaptive_step_count": 0,
            "tool_call_count": 0,
            "evidence_sufficient": False,
            "evidence_records": [],
            "trace_id": trace_id,
            "request_id": trace_id or uuid.uuid4().hex,
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
                route = self.nodes.route_after_retrieval(state)
                if route == "adaptive":
                    state.update(await self.nodes.adaptive_read(state))
                elif route == "tools":
                    state.update(await self.nodes.execute_tools(state))
                state.update(await self.nodes.answer(state))
            else:
                with context:
                    state.update(await self.nodes.plan(state))
                    state.update(await self.nodes.retrieve_knowledge(state))
                    route = self.nodes.route_after_retrieval(state)
                    if route == "adaptive":
                        state.update(await self.nodes.adaptive_read(state))
                    elif route == "tools":
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
        builder.add_node("adaptive", self.nodes.adaptive_read)
        builder.add_node("answer", self.nodes.answer)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan", self.nodes.route_after_plan, {"tools": "retrieve", "answer": "retrieve"}
        )
        builder.add_conditional_edges(
            "retrieve", self.nodes.route_after_retrieval, {"adaptive": "adaptive", "tools": "tools", "answer": "answer"}
        )
        builder.add_edge("adaptive", "answer")
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
    max_steps: int = 8,
    max_identical_tool_calls: int = 2,
    max_consecutive_tool_failures: int = 2,
    knowledge_retriever=None,
    knowledge_top_k: int = 5,
    task_planning_service=None,
    approval_store=None,
    action_verifier=None,
    trace_recorder=None,
    autonomy_enabled: bool = False,
    auto_actions_per_request: int = 1,
    auto_resume_max_datasets: int = 3,
):
    policy = AgentPolicyEngine(max_tool_calls=max_tool_calls)
    autonomy_policy = BoundedAutonomyPolicy(
        enabled=autonomy_enabled,
        max_actions_per_request=auto_actions_per_request,
        max_resume_datasets=auto_resume_max_datasets,
    )
    action_coordinator = (
        WriteActionCoordinator(
            tool_client,
            policy,
            approval_store,
            verifier=action_verifier,
            trace_recorder=trace_recorder,
            autonomy_policy=autonomy_policy,
        )
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
        max_steps=max_steps,
        max_identical_tool_calls=max_identical_tool_calls,
        max_consecutive_tool_failures=max_consecutive_tool_failures,
        autonomy_policy=autonomy_policy,
    )
    runtime = (runtime or "langgraph").strip().lower()
    if runtime in {"sequential", "test"}:
        return SequentialReadOnlyAgent(nodes, conversation_store, action_coordinator, trace_recorder=trace_recorder)
    if runtime == "langgraph":
        return LangGraphReadOnlyAgent(nodes, conversation_store, action_coordinator, trace_recorder=trace_recorder)
    raise ValueError(f"Unsupported PLATFORM_AGENT_RUNTIME: {runtime}")
