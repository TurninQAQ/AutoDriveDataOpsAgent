from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .approval import ApprovalStore, PendingApproval
from .autonomy import AutonomyDecision, AutonomyMode, BoundedAutonomyPolicy
from .goal_verification import GoalVerifier
from .models import AgentIntent, AgentPlan, ToolCallSpec, ToolObservation
from .policy import AgentPolicyEngine
from .verification import ActionVerifier


WRITE_INTENT_TO_TOOL = {
    AgentIntent.SUBMIT_TASK: "submit_task",
    AgentIntent.RESUME_TASK: "resume_task",
    AgentIntent.SET_TASK_PRIORITY: "set_task_priority",
    AgentIntent.STOP_TASK: "stop_task",
    AgentIntent.DELETE_TASK: "delete_task",
}


@dataclass
class PreparedWriteAction:
    tool_name: str
    arguments: dict[str, Any]
    precondition: dict[str, Any]
    verification_baseline: dict[str, Any]
    impact_summary: str
    impact_details: list[str]
    risk_level: str


class WriteActionCoordinator:
    def __init__(
        self,
        tool_client,
        policy: AgentPolicyEngine,
        approval_store: ApprovalStore,
        verifier: ActionVerifier | None = None,
        goal_verifier: GoalVerifier | None = None,
        trace_recorder=None,
        autonomy_policy: BoundedAutonomyPolicy | None = None,
    ):
        self.tool_client = tool_client
        self.policy = policy
        self.approval_store = approval_store
        self.verifier = verifier or ActionVerifier(tool_client)
        self.goal_verifier = goal_verifier or GoalVerifier(tool_client)
        self.trace_recorder = trace_recorder
        self.autonomy_policy = autonomy_policy

    @staticmethod
    def _find_observation(observations: list[ToolObservation], tool_name: str) -> ToolObservation | None:
        for item in observations:
            if item.tool_name == tool_name and item.ok:
                return item
        return None

    def impact(self, plan: AgentPlan, observations: list[ToolObservation], arguments: dict[str, Any]) -> tuple[str, list[str]]:
        action = WRITE_INTENT_TO_TOOL[plan.intent]
        queue_obs = self._find_observation(observations, "get_queue_state")
        queue = queue_obs.data if queue_obs and isinstance(queue_obs.data, dict) else {}
        active = queue.get("active") or {}
        active_name = str(active.get("task_name") or "")
        active_priority = active.get("priority")
        details: list[str] = []

        if action == "submit_task":
            config = arguments.get("config") or {}
            priority = config.get("priority")
            task_type = config.get("task_type") or ""
            details.append(f"Task prefix={arguments.get('task_prefix')} task_type={task_type or '(default)'} priority={priority if priority is not None else '(resolved by config)'}.")
            details.append(f"Datasets={len(config.get('datasets') or [])}; max_active_runs={config.get('max_active_runs')}.")
            if active_name:
                details.append(f"Current active task={active_name} priority={active_priority}; submit may queue or request soft preemption according to platform priority rules.")
            return "Submit a new business task and create/trigger its Airflow DagRuns.", details

        task_name = str(arguments.get("task_name") or plan.task_name or "")
        if action == "set_task_priority":
            new_priority = arguments.get("priority")
            details.append(f"Task {task_name} priority will change to {new_priority}.")
            if active_name:
                details.append(f"Current active task={active_name} priority={active_priority}; queue refresh may move a task into draining/preemption flow.")
            return "Change business-task priority and refresh the global priority queue.", details
        if action == "resume_task":
            datasets = arguments.get("datasets") or []
            details.append(f"Task={task_name}; datasets={','.join(datasets) if datasets else 'failed datasets selected by platform'}.")
            if active_name and active_name != task_name:
                details.append(f"Current active task={active_name}; resumed task may remain queued or request soft preemption depending on priority.")
            return "Create new DagRuns for failed or selected datasets and re-enter platform scheduling.", details
        if action == "stop_task":
            datasets = arguments.get("datasets") or []
            if datasets:
                details.append(f"Only selected datasets will be stopped: {','.join(datasets)}.")
            else:
                details.append("The entire task will be stopped; active DagRuns may be marked failed, containers stopped and queue entry removed.")
            if active_name == task_name:
                details.append("The target is currently active; stopping the entire task can advance the next queued task.")
            return "Stop active processing and reclaim task-owned runtime resources.", details
        if action == "delete_task":
            details.extend([
                f"Task={task_name}.",
                "Generated DAG/config will be deleted after active runs/containers/reservations are cleaned.",
                "This operation is destructive and cannot be undone by the Agent.",
            ])
            return "Permanently delete the generated business task and its platform metadata/artifacts.", details
        return action, details

    async def _build_context(
        self,
        *,
        state_user_text: str,
        thread_id: str,
        plan: AgentPlan,
        observations: list[ToolObservation],
        task_plan: dict[str, Any] | None = None,
    ) -> PreparedWriteAction:
        tool_name = WRITE_INTENT_TO_TOOL.get(plan.intent)
        if not tool_name:
            raise ValueError(f"Intent is not a write action: {plan.intent}")
        self.policy.validate_write_tool(tool_name)
        arguments = dict(plan.write_action or {})

        if tool_name == "submit_task":
            if not task_plan or not task_plan.get("valid"):
                raise ValueError("submit_task requires a valid V0.6 TaskPlanningResult")
            spec = task_plan.get("task_spec") or {}
            task_prefix = str(spec.get("task_prefix") or "")
            config = task_plan.get("config") or {}
            validation = await self.tool_client.execute([
                ToolCallSpec(name="validate_task_spec", arguments={"task_prefix": task_prefix, "config": config})
            ])
            if not validation or not validation[0].ok:
                error = validation[0].error if validation else "validate_task_spec returned no result"
                raise ValueError(f"TaskSpec write-boundary validation failed: {error}")
            arguments = {"task_prefix": task_prefix, "config": config}
            pre_task_name = ""
        else:
            pre_task_name = str(arguments.get("task_name") or plan.task_name or "")
            if not pre_task_name:
                raise ValueError(f"{tool_name} requires task_name")
            arguments["task_name"] = pre_task_name

        pre = await self.tool_client.execute([
            ToolCallSpec(name="get_write_precondition", arguments={"task_name": pre_task_name})
        ])
        if not pre or not pre[0].ok or not isinstance(pre[0].data, dict):
            error = pre[0].error if pre else "get_write_precondition returned no result"
            raise RuntimeError(f"Failed to capture write precondition: {error}")

        verification_baseline: dict[str, Any] = {}
        if tool_name != "submit_task":
            baseline = await self.tool_client.execute([
                ToolCallSpec(name="get_action_verification_snapshot", arguments={
                    "task_name": pre_task_name, "datasets": list(arguments.get("datasets") or []), "airflow_limit": 200,
                })
            ])
            if baseline and baseline[0].ok and isinstance(baseline[0].data, dict):
                verification_baseline = baseline[0].data

        impact_summary, impact_details = self.impact(plan, observations, arguments)
        risk = self.policy.risk_for_tool(tool_name)
        return PreparedWriteAction(
            tool_name=tool_name,
            arguments=arguments,
            precondition=dict(pre[0].data),
            verification_baseline=verification_baseline,
            impact_summary=impact_summary,
            impact_details=impact_details,
            risk_level=risk,
        )

    def _create_hitl_approval(
        self,
        *,
        context: PreparedWriteAction,
        state_user_text: str,
        thread_id: str,
        trace_id: str,
        policy_decision: AutonomyDecision | None = None,
    ) -> PendingApproval:
        pending = self.approval_store.create(
            thread_id=thread_id,
            user_request=state_user_text,
            tool_name=context.tool_name,
            arguments=context.arguments,
            precondition=context.precondition,
            risk_level=context.risk_level,
            impact_summary=context.impact_summary,
            impact_details=context.impact_details,
            verification_baseline=context.verification_baseline,
            trace_id=trace_id,
            policy_decision=policy_decision.model_dump(mode="json") if policy_decision else None,
        )
        if self.trace_recorder is not None and trace_id:
            self.trace_recorder.record(
                trace_id,
                "approval",
                "approval_created",
                status="pending",
                data={
                    "approval_id": pending.approval_id,
                    "tool": context.tool_name,
                    "risk_level": context.risk_level,
                    "arguments": context.arguments,
                    "precondition": context.precondition,
                    "impact_summary": context.impact_summary,
                },
            )
        return pending

    async def prepare(
        self,
        *,
        state_user_text: str,
        thread_id: str,
        plan: AgentPlan,
        observations: list[ToolObservation],
        task_plan: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> PendingApproval:
        context = await self._build_context(
            state_user_text=state_user_text,
            thread_id=thread_id,
            plan=plan,
            observations=observations,
            task_plan=task_plan,
        )
        return self._create_hitl_approval(
            context=context,
            state_user_text=state_user_text,
            thread_id=thread_id,
            trace_id=trace_id,
        )

    async def prepare_with_autonomy(
        self,
        *,
        state_user_text: str,
        thread_id: str,
        plan: AgentPlan,
        observations: list[ToolObservation],
        task_plan: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> tuple[AutonomyDecision, PendingApproval | None]:
        """Prepare a frozen action and authorize it through the V1.7 policy."""

        context = await self._build_context(
            state_user_text=state_user_text,
            thread_id=thread_id,
            plan=plan,
            observations=observations,
            task_plan=task_plan,
        )
        if self.autonomy_policy is None:
            decision = AutonomyDecision(
                mode=AutonomyMode.HITL,
                action=context.tool_name,
                risk_level=context.risk_level,
                eligible=False,
                reasons=["autonomy_policy_unavailable"],
                frozen_arguments=context.arguments,
            )
        else:
            decision = self.autonomy_policy.decide(
                action=context.tool_name,
                arguments=context.arguments,
                precondition=context.precondition,
                baseline=context.verification_baseline,
                auto_actions_used=self.approval_store.count_auto_actions(trace_id),
            )

        if decision.mode == AutonomyMode.AUTO:
            context.arguments = dict(decision.frozen_arguments)
            context.impact_summary, context.impact_details = self.impact(plan, observations, context.arguments)
            auto = self.approval_store.create_auto_execution(
                thread_id=thread_id,
                user_request=state_user_text,
                tool_name=context.tool_name,
                arguments=context.arguments,
                precondition=context.precondition,
                risk_level=context.risk_level,
                impact_summary=context.impact_summary,
                impact_details=context.impact_details,
                verification_baseline=context.verification_baseline,
                trace_id=trace_id,
                policy_decision=decision.model_dump(mode="json"),
            )
            return decision, auto

        if decision.mode == AutonomyMode.DENY:
            return decision, None

        pending = self._create_hitl_approval(
            context=context,
            state_user_text=state_user_text,
            thread_id=thread_id,
            trace_id=trace_id,
            policy_decision=decision,
        )
        return decision, pending

    async def execute_approval(self, approval_id: str, execution_trace_id: str = "") -> PendingApproval:
        preview = self.approval_store.get(approval_id)
        self.policy.validate_write_tool(preview.tool_name)
        if preview.authorization_mode == "auto":
            if preview.status != "executing":
                raise RuntimeError(f"Auto action is not executable from status: {preview.status}")
            item = preview
        else:
            item = self.approval_store.claim_for_execution(approval_id, execution_trace_id=execution_trace_id)
        if self.trace_recorder is not None and execution_trace_id:
            self.trace_recorder.record(
                execution_trace_id,
                "approval",
                "approval_claimed",
                status="ok",
                data={
                    "approval_id": item.approval_id,
                    "origin_trace_id": item.trace_id,
                    "tool": item.tool_name,
                    "arguments": item.arguments,
                    "risk_level": item.risk_level,
                },
                parent_trace_id=item.trace_id or None,
            )
        arguments = dict(item.arguments)
        arguments["precondition"] = dict(item.precondition)
        try:
            observations = await self.tool_client.execute([
                ToolCallSpec(name=item.tool_name, arguments=arguments)
            ])
        except Exception as exc:
            failed = self.approval_store.mark_failed(approval_id, str(exc))
            if self.trace_recorder is not None and execution_trace_id:
                self.trace_recorder.record(execution_trace_id, "mutation", item.tool_name, status="error", data={"approval_id": approval_id, "error": str(exc)})
                if item.authorization_mode == "auto":
                    self.trace_recorder.record(execution_trace_id, "autonomy", "failure_escalation", status="error", data={"approval_id": approval_id, "reason": "mutation_failure", "automatic_retry": False})
            return failed
        if not observations:
            failed = self.approval_store.mark_failed(approval_id, "Write MCP tool returned no observation")
            if self.trace_recorder is not None and execution_trace_id:
                self.trace_recorder.record(execution_trace_id, "mutation", item.tool_name, status="error", data={"approval_id": approval_id, "error": failed.error})
                if item.authorization_mode == "auto":
                    self.trace_recorder.record(execution_trace_id, "autonomy", "failure_escalation", status="error", data={"approval_id": approval_id, "reason": "mutation_failure", "automatic_retry": False})
            return failed
        observation = observations[0]
        if not observation.ok:
            failed = self.approval_store.mark_failed(approval_id, observation.error or "Write MCP tool failed")
            if self.trace_recorder is not None and execution_trace_id:
                self.trace_recorder.record(execution_trace_id, "mutation", item.tool_name, status="error", data={"approval_id": approval_id, "error": failed.error})
                if item.authorization_mode == "auto":
                    self.trace_recorder.record(execution_trace_id, "autonomy", "failure_escalation", status="error", data={"approval_id": approval_id, "reason": "mutation_failure", "automatic_retry": False})
            return failed
        result = observation.data if isinstance(observation.data, dict) else {"data": observation.data}
        verification = await self.verifier.verify(action=item.tool_name, arguments=item.arguments, execution_result=result, baseline=item.verification_baseline)
        payload = verification.model_dump(mode="json")
        if self.trace_recorder is not None and execution_trace_id:
            self.trace_recorder.record(
                execution_trace_id,
                "verification",
                item.tool_name,
                status="verified" if verification.verified else verification.status,
                data={"approval_id": approval_id, "result": payload},
            )
        if verification.verified:
            goal_payload = None
            if item.tool_name == "resume_task":
                goal_verification = await self.goal_verifier.verify_resume(
                    arguments=item.arguments,
                    action_verification=verification,
                    baseline=item.verification_baseline,
                )
                goal_payload = goal_verification.model_dump(mode="json")
                if self.trace_recorder is not None and execution_trace_id:
                    self.trace_recorder.record(
                        execution_trace_id,
                        "goal_verification",
                        item.tool_name,
                        status=goal_verification.status,
                        data={"approval_id": approval_id, "result": goal_payload},
                    )
                    if item.authorization_mode == "auto" and goal_verification.status != "satisfied":
                        self.trace_recorder.record(
                            execution_trace_id,
                            "autonomy",
                            "failure_escalation",
                            status="escalate",
                            data={
                                "approval_id": approval_id,
                                "reason": f"goal_verification_{goal_verification.status}",
                                "automatic_retry": False,
                            },
                        )
            executed = self.approval_store.mark_executed(
                approval_id,
                result,
                verification_result=payload,
                goal_verification_result=goal_payload,
            )
            if self.trace_recorder is not None and execution_trace_id:
                self.trace_recorder.record(execution_trace_id, "approval", "approval_executed", status="executed", data={"approval_id": approval_id, "tool": item.tool_name})
            return executed
        failed_names = [check.name for check in verification.checks if not check.passed]
        detail = ",".join(failed_names) or verification.status
        failed = self.approval_store.mark_verification_failed(approval_id, result, payload, f"Action executed but verification {verification.status}: {detail}")
        if self.trace_recorder is not None and execution_trace_id:
            self.trace_recorder.record(execution_trace_id, "approval", "approval_verification_failed", status="verification_failed", data={"approval_id": approval_id, "tool": item.tool_name, "error": failed.error})
            if item.authorization_mode == "auto":
                self.trace_recorder.record(execution_trace_id, "autonomy", "failure_escalation", status="error", data={"approval_id": approval_id, "reason": "action_verification_failure", "automatic_retry": False})
        return failed
