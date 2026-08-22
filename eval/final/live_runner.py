"""Live-evaluation adapters for the frozen Agent runtime.

The module deliberately separates three things:

* ``ScenarioExecutionInput`` is the only benchmark data visible to an Agent.
* ``ScenarioGroundTruth`` is evaluator-only and is never passed to a runner.
* scripted runners remain in :mod:`formal_runners`; these runners use a model
  client and the real sequential Agent runtime when a client is supplied.

The readiness gate uses ``FakeLLMClient`` in tests.  No provider is contacted
unless a caller explicitly supplies a provider-backed model client factory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from platform_agent.approval import ApprovalStore
from platform_agent.memory import ConversationStore
from platform_agent.models import AgentIntent, AgentPlan, AgentResponse, AgentStepDecision, AgentStepAction, ToolCallSpec, ToolObservation
from platform_agent.workflow import build_agent_runtime
from platform_planning.service import TaskPlanningService

from .fixture_registry import Fixture, resolve_fixture
from .schema import Scenario, load_scenarios
from .collector import (
    CollectorConfig,
    QuotaBlockedError,
    adapter_for,
    collect_trajectories_with_status,
    prepare_run_directory,
    translate_provider_exception,
    write_raw_trajectories,
)
from .runner import EVALUATOR_VERSION, run_evaluation
from .schema import file_sha256
from .telemetry import InstrumentedModelClient, ModelTelemetry


LIVE_RUNNER_VERSION = "a-plus-final-live-runner-v3"


@dataclass(frozen=True)
class ScenarioExecutionInput:
    """Runtime-visible scenario data; no expected_* fields are present."""

    case_id: str
    prompt: str
    fixture: str | None
    fixture_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioGroundTruth:
    """Evaluator-only truth kept separate from the execution input."""

    case_id: str
    expected_intent: str
    expected_target: str | None
    expected_policy: str | None
    expected_goal: str | None
    expected_datasets: tuple[str, ...] = ()


def provider_model_factory(model_name: str, _execution_input: ScenarioExecutionInput):
    """Build the configured real adapter lazily; construction performs no call."""
    from platform_agent.model import build_model_from_env
    from platform_agent.settings import AgentSettings

    settings = AgentSettings.from_env()
    return build_model_from_env(
        settings.provider,
        model_name,
        settings.temperature,
        request_timeout_sec=settings.request_timeout_sec,
    )


def execution_input_for(scenario: Scenario) -> ScenarioExecutionInput:
    return ScenarioExecutionInput(
        case_id=scenario.id,
        prompt=scenario.prompt,
        fixture=scenario.fixture if scenario.live_fixture_required else None,
        fixture_payload=dict(scenario.fixture_payload),
    )


def ground_truth_for(scenario: Scenario) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        case_id=scenario.id,
        expected_intent=scenario.expected_intent,
        expected_target=scenario.expected_target,
        expected_policy=scenario.expected_policy,
        expected_goal=scenario.expected_goal,
        expected_datasets=tuple(scenario.expected_datasets),
    )


def assert_ground_truth_isolated(execution_input: ScenarioExecutionInput) -> None:
    names = set(vars(execution_input))
    leaked = sorted(name for name in names if name.startswith("expected_"))
    if leaked:
        raise AssertionError(f"ground truth leaked into Agent execution input: {leaked}")


def validate_live_fixtures(cases: list[Scenario]) -> int:
    """Construct every required isolated runtime before a live run starts."""
    count = 0
    for scenario in cases:
        if not scenario.live_fixture_required:
            continue
        fixture = resolve_fixture(scenario.fixture)
        FixtureToolClient(fixture)
        count += 1
    return count


class FixtureToolClient:
    """Small MCP-shaped deterministic runtime for live readiness tests."""

    def __init__(self, fixture: Fixture):
        self.fixture = fixture
        self.calls: list[ToolCallSpec] = []
        self.mutation_calls = 0
        self.snapshot_calls = 0
        self._new_runs: list[dict[str, Any]] = []
        self.observations: list[ToolObservation] = []
        self._task_exists = fixture.task_exists
        self._config_exists = fixture.config_exists
        self._dag_exists = fixture.dag_exists
        self._priority: int | None = None
        self._stopped = False
        self._submitted = False
        self._deleted = False

    async def describe_tools(self) -> list[dict[str, Any]]:
        from platform_agent.tool_catalog import build_read_only_tool_catalog

        return build_read_only_tool_catalog(knowledge_enabled=True) + [
            {"name": name, "description": name, "input_schema": {}}
            for name in ("get_write_precondition", "get_action_verification_snapshot", "resume_task", "submit_task", "stop_task", "delete_task", "set_task_priority")
        ]

    def _runs(self) -> list[dict[str, Any]]:
        if self._deleted:
            return []
        runs: list[dict[str, Any]] = []
        for dataset, state in self.fixture.latest_dataset_states.items():
            runs.append({"run_id": f"old-{dataset}", "dataset_name": dataset, "state": "stopped" if self._stopped and state in {"queued", "running", "scheduled"} else state})
        return runs + list(self._new_runs)

    def _snapshot(self, task_name: str) -> dict[str, Any]:
        observed_task = "other_task" if self.fixture.provenance_conflict else task_name
        queue_active = not self._deleted and not self._stopped and (
            self.fixture.task_exclusive or self._submitted or self.fixture.active_task_name
        )
        queue_entry = None
        if queue_active:
            queue_entry = {"task_name": task_name}
            if self._priority is not None:
                queue_entry["priority"] = self._priority
        return {
            "task_name": observed_task,
            "task_exists": self._task_exists,
            "config_file_exists": self._config_exists,
            "dag_file_exists": self._dag_exists,
            "airflow_dag_exists": self._dag_exists,
            "available_datasets": list(self.fixture.available_datasets),
            "task_exclusive": self.fixture.task_exclusive,
            "queue": {
                "location": "queued" if queue_active else "not_found",
                "position": 0 if queue_active else -1,
                "entry": queue_entry,
            },
            "active_task_name": "" if self._stopped or self._deleted else (self.fixture.active_task_name or (task_name if self._submitted else "")),
            "priority": self._priority,
            "containers": [],
            "gpu_reservations": [],
            "airflow_runs": self._runs(),
            "errors": {} if self.fixture.critical_evidence_available else {"airflow": "backend unavailable"},
        }

    async def execute(self, calls: list[ToolCallSpec]) -> list[ToolObservation]:
        result: list[ToolObservation] = []
        for call in calls:
            self.calls.append(call)
            name = call.name
            args = dict(call.arguments)
            if name == "get_task_detail":
                data = {
                    "task_name": args.get("task_name"),
                    "task_exists": self._task_exists,
                    "config_file_exists": self._config_exists,
                    "dag_file_exists": self._dag_exists,
                    "datasets": list(self.fixture.available_datasets),
                    "state": "stopped" if self._stopped else next(iter(self.fixture.latest_dataset_states.values()), "unknown"),
                    "task_state": "stopped" if self._stopped else next(iter(self.fixture.latest_dataset_states.values()), "unknown"),
                    "priority": self._priority,
                }
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data=data))
            elif name == "get_queue_state":
                active = None
                if self.fixture.active_task_name and not self._stopped and not self._deleted:
                    active = {"task_name": self.fixture.active_task_name, "priority": self._priority or 1}
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"version": 1, "active": active, "queue": []}))
            elif name == "get_gpu_pool":
                state = "degraded" if "gpu" in self.fixture.name else "healthy"
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"gpu_state": state, "devices": []}))
            elif name == "get_stage_logs":
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"stage": "precheck", "error": "PRECHECK_FAILED"}))
            elif name == "diagnose_task":
                facts = self._diagnosis_facts()
                if not self.fixture.critical_evidence_available:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=False, error="backend unavailable"))
                else:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={**facts, "task_name": args.get("task_name")}))
            elif name == "search_knowledge":
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"results": [{"title": "runbook", "content": "soft preemption"}]}))
            elif name == "get_write_precondition":
                if not self.fixture.critical_evidence_available:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=False, error="critical evidence unavailable"))
                else:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={
                        "queue_sha256": "queue-1",
                        "task_name": args.get("task_name") or "",
                        "task_config_sha256": "config-1",
                        "task_exists": self._task_exists,
                        "active_task_name": self.fixture.active_task_name or "",
                    }))
            elif name == "get_action_verification_snapshot":
                self.snapshot_calls += 1
                if not self.fixture.critical_evidence_available:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=False, error="critical evidence unavailable"))
                else:
                    result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data=self._snapshot(str(args.get("task_name") or ""))))
            elif name == "resume_task":
                self.mutation_calls += 1
                datasets = [str(item) for item in args.get("datasets") or [] if str(item)]
                for dataset in datasets or list(self.fixture.currently_failed_datasets):
                    self._new_runs.append({"run_id": f"new-{self.mutation_calls}-{dataset}", "dataset_name": dataset, "state": self.fixture.post_goal.lower() if self.fixture.post_goal else "running"})
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"ok": True, "task_name": args.get("task_name"), "datasets": datasets}))
            elif name == "stop_task":
                self.mutation_calls += 1
                self._stopped = True
                for run in self._runs():
                    if str(run.get("state") or "").lower() in {"queued", "running", "scheduled"}:
                        run["state"] = "stopped"
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"ok": True, "task_name": args.get("task_name")}))
            elif name == "submit_task":
                self.mutation_calls += 1
                self._submitted = True
                self._task_exists = True
                self._config_exists = True
                self._dag_exists = True
                config = args.get("config") if isinstance(args.get("config"), Mapping) else {}
                task_name_value = str(args.get("task_name") or config.get("task_prefix") or self.fixture.task_name or "")
                datasets = [
                    str(item.get("dataset_name") or item.get("dataset_path") or "")
                    for item in config.get("datasets") or []
                    if isinstance(item, Mapping)
                ]
                for dataset in datasets:
                    self._new_runs.append({"run_id": f"submitted-{self.mutation_calls}-{dataset}", "dataset_name": dataset, "state": "queued"})
                priority = config.get("priority")
                if priority is not None:
                    self._priority = int(priority)
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"ok": True, "task_name": task_name_value, "triggered": len(datasets)}))
            elif name == "delete_task":
                self.mutation_calls += 1
                self._deleted = True
                self._task_exists = False
                self._config_exists = False
                self._dag_exists = False
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"ok": True, "task_name": args.get("task_name")}))
            elif name == "set_task_priority":
                self.mutation_calls += 1
                self._priority = int(args.get("priority"))
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={"ok": True, "task_name": args.get("task_name"), "priority": self._priority}))
            else:
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data={}))
        self.observations.extend(result)
        return result

    def _diagnosis_facts(self) -> dict[str, Any]:
        name = self.fixture.name
        if "draining" in name:
            return {"task_state": "draining", "reason_code": "SOFT_PREEMPTION"}
        if "partial_multi" in name:
            return {"task_state": "in_progress", "datasets": ["A", "B"]}
        if "backend_failure" in name:
            return {"status": "inconclusive"}
        if "wrong_task" in name:
            return {"status": "inconclusive", "task_name": "other_task"}
        return {"task_state": "failed", "dataset": "A", "failed_stage": "precheck", "reason_code": "PRECHECK_FAILED"}


@dataclass
class FakeLLMClient:
    """Provider-neutral model stub used only by live integration tests."""

    plan_result: AgentPlan
    structured_facts: dict[str, Any] = field(default_factory=dict)
    structured_plan: dict[str, Any] = field(default_factory=dict)
    refusal: bool = False
    requires_tool_descriptions: bool = True
    last_plan: AgentPlan | None = None

    async def plan(self, user_text, tool_descriptions, history):
        self.last_plan = self.plan_result
        return self.plan_result

    async def synthesize(self, user_text, plan, observations, history, knowledge=None):
        response = AgentResponse(
            intent=plan.intent,
            summary="deterministic fake model response",
            confidence="high",
            evidence=[json.dumps(self.structured_facts, sort_keys=True)] if self.structured_facts else [],
        )
        return response


def _build_agent(execution_input: ScenarioExecutionInput, model: Any, *, autonomy_enabled: bool, root: Path):
    fixture = resolve_fixture(execution_input.fixture or "")
    client = FixtureToolClient(fixture)
    store = ApprovalStore(root / "approvals")
    agent = build_agent_runtime(
        "sequential",
        model,
        client,
        ConversationStore(root / "sessions"),
        task_planning_service=TaskPlanningService(),
        approval_store=store,
        autonomy_enabled=autonomy_enabled,
        auto_actions_per_request=1,
        auto_resume_max_datasets=3,
    )
    agent.nodes.action_coordinator.verifier.attempts = 1
    agent.nodes.action_coordinator.verifier.interval_sec = 0
    agent.nodes.action_coordinator.goal_verifier.attempts = 1
    agent.nodes.action_coordinator.goal_verifier.interval_sec = 0
    return agent, client, store


def _runtime_structured_facts(response: AgentResponse, client: FixtureToolClient) -> dict[str, Any]:
    """Extract facts from observations/AgentResponse, never benchmark truth."""
    for observation in reversed(client.observations):
        if observation.tool_name in {"diagnose_task", "get_task_detail", "get_gpu_pool", "get_stage_logs"} and isinstance(observation.data, Mapping):
            facts = {
                str(key): value
                for key, value in observation.data.items()
                if key not in {"task_name", "task_exists", "config_file_exists", "dag_file_exists", "datasets"}
            }
            if facts:
                return facts
    for item in response.evidence:
        if not isinstance(item, str):
            continue
        try:
            value = json.loads(item)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _runtime_structured_plan(response: AgentResponse) -> dict[str, Any]:
    task_plan = response.task_plan if isinstance(response.task_plan, Mapping) else {}
    task_spec = task_plan.get("task_spec") or {}
    if isinstance(task_spec, Mapping) and task_spec:
        return {
            "task_name": task_spec.get("task_prefix"),
            "datasets": [item.get("dataset_path") for item in task_spec.get("datasets") or [] if isinstance(item, Mapping)],
            "stages": task_spec.get("pipeline_stages") or [],
            "priority": task_plan.get("resolved_priority", task_spec.get("priority")),
        }
    initial = response.initial_plan if isinstance(response.initial_plan, Mapping) else {}
    draft = initial.get("task_draft") if isinstance(initial.get("task_draft"), Mapping) else {}
    if draft:
        return {
            "task_name": draft.get("task_prefix") or initial.get("task_name"),
            "datasets": draft.get("dataset_paths") or draft.get("dataset_names") or [],
            "stages": draft.get("pipeline_stages") or [],
            "priority": draft.get("priority"),
        }
    return {}


def _raw_response(response: AgentResponse, client: FixtureToolClient, model: Any, *, approval_count: int = 0, mutation_count_before_approval: int = 0) -> dict[str, Any]:
    pending = response.pending_action or {}
    arguments = pending.get("arguments") if isinstance(pending, Mapping) else {}
    arguments = arguments if isinstance(arguments, Mapping) else {}
    structured_facts = _runtime_structured_facts(response, client)
    # Compatibility fallback for the pre-existing scripted test double only.
    # Real provider objects are never required to define these attributes.
    if not structured_facts:
        candidate = getattr(model, "structured_facts", {})
        if isinstance(candidate, Mapping):
            structured_facts = dict(candidate)
    structured_plan = _runtime_structured_plan(response)
    if not structured_plan:
        candidate = getattr(model, "structured_plan", {})
        if isinstance(candidate, Mapping):
            structured_plan = dict(candidate)
    policy_mode = (response.policy_decision or {}).get("mode")
    explicit_refusal = bool(
        response.blocked
        and not response.approval_required
        and not response.action_result
        and policy_mode in {None, "DENY"}
    )
    if not explicit_refusal:
        explicit_refusal = bool(getattr(model, "refusal", False))
    return {
        "intent": response.intent.value,
        "target": arguments.get("task_name") or (response.initial_plan or {}).get("task_name"),
        "policy_mode": (response.policy_decision or {}).get("mode"),
        "authorization_mode": response.authorization_mode,
        "frozen_arguments": dict(arguments),
        # B1 intentionally retains the V1.6 legacy empty-dataset argument;
        # record the deterministic scope actually selected by the fixture as
        # the business execution scope while preserving frozen_arguments.
        "frozen_datasets": list(arguments.get("datasets") or client.fixture.currently_failed_datasets),
        "approval_required": response.approval_required,
        "approval_created": bool(response.approval_id),
        "approval_count": approval_count,
        "oracle_approval": approval_count > 0,
        "mutation_count_before_approval": mutation_count_before_approval,
        "mutation_count": client.mutation_calls,
        "action_verification": response.action_verification,
        "goal_verification": response.goal_verification_result,
        "predicted_goal": (response.goal_verification_result or {}).get("status") or (response.goal_progress.value if response.goal_progress else None),
        "tool_calls": [call.name for call in client.calls],
        "structured_facts": structured_facts,
        "structured_plan": structured_plan,
        "direct_write": False,
        "direct_model_write": False,
        "adaptive_write": 0,
        "sandbox_only": False,
        "explicit_refusal": explicit_refusal,
        "refusal": explicit_refusal,
    }


def _finish_telemetry(raw: dict[str, Any], telemetry: ModelTelemetry, started: float) -> dict[str, Any]:
    raw.update(telemetry.as_dict())
    raw["attempt_wall_latency_ms"] = round(max(0.0, (time.perf_counter() - started) * 1000), 3)
    raw["latency_ms"] = raw["attempt_wall_latency_ms"]
    return raw


def _attach_telemetry(error: BaseException, telemetry: ModelTelemetry, started: float) -> BaseException:
    # Exception metadata is deliberately limited to safe counters/codes.
    setattr(error, "telemetry", telemetry.as_dict())
    setattr(error, "attempt_wall_latency_ms", round(max(0.0, (time.perf_counter() - started) * 1000), 3))
    return error


class LiveFullRunner:
    system = "full"

    def __init__(self, model_factory: Callable[[str, ScenarioExecutionInput], Any] | None = None):
        self.model_factory = model_factory or provider_model_factory

    async def run(self, execution_input: ScenarioExecutionInput, model_client: Any, *, root: Path | None = None, model_name: str = "live-model", oracle_approval: bool = True) -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        started = time.perf_counter()
        telemetry = ModelTelemetry()
        instrumented_model = InstrumentedModelClient(model_client, telemetry)
        work_root = root or Path(tempfile.mkdtemp(prefix="a-plus-live-full-"))
        agent, client, _store = _build_agent(execution_input, instrumented_model, autonomy_enabled=True, root=work_root)
        try:
            response = await agent.run(execution_input.prompt, f"live-{execution_input.case_id}")
        except Exception as exc:
            translated = translate_provider_exception(exc, model=model_name, free_tier_only=True)
            if isinstance(translated, QuotaBlockedError):
                raise _attach_telemetry(translated, telemetry, started)
            raise _attach_telemetry(translated, telemetry, started)
        if oracle_approval and response.approval_required and response.authorization_mode == "hitl" and (response.policy_decision or {}).get("mode") == "HITL":
            before = client.mutation_calls
            item = await agent.approve(response.approval_id)
            response.action_verification = item.verification_result
            response.goal_verification_result = item.goal_verification_result
            response.pending_action = item.model_dump(mode="json")
            response.action_result = item.execution_result
            return _finish_telemetry(_raw_response(response, client, model_client, approval_count=1, mutation_count_before_approval=before), telemetry, started)
        return _finish_telemetry(_raw_response(response, client, model_client), telemetry, started)

    def __call__(self, scenario: Scenario, repetition: int, model: str) -> Mapping[str, Any]:
        if self.model_factory is None:
            raise RuntimeError("LiveFullRunner requires an explicit provider/model factory")
        execution_input = execution_input_for(scenario)
        client = self.model_factory(model, execution_input)
        return asyncio.run(self.run(execution_input, client, model_name=model))


class LiveHitlOnlyRunner(LiveFullRunner):
    system = "hitl_only"

    async def run(self, execution_input: ScenarioExecutionInput, model_client: Any, *, root: Path | None = None, model_name: str = "live-model") -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        started = time.perf_counter()
        telemetry = ModelTelemetry()
        instrumented_model = InstrumentedModelClient(model_client, telemetry)
        work_root = root or Path(tempfile.mkdtemp(prefix="a-plus-live-hitl-"))
        agent, client, _store = _build_agent(execution_input, instrumented_model, autonomy_enabled=False, root=work_root)
        try:
            response = await agent.run(execution_input.prompt, f"live-{execution_input.case_id}")
        except Exception as exc:
            translated = translate_provider_exception(exc, model=model_name, free_tier_only=True)
            if isinstance(translated, QuotaBlockedError):
                raise _attach_telemetry(translated, telemetry, started)
            raise _attach_telemetry(translated, telemetry, started)
        if response.approval_required and response.authorization_mode == "hitl" and (response.policy_decision or {}).get("mode") == "HITL":
            before = client.mutation_calls
            item = await agent.approve(response.approval_id)
            response.action_verification = item.verification_result
            response.goal_verification_result = item.goal_verification_result
            response.pending_action = item.model_dump(mode="json")
            response.action_result = item.execution_result
            return _finish_telemetry(_raw_response(response, client, model_client, approval_count=1, mutation_count_before_approval=before), telemetry, started)
        return _finish_telemetry(_raw_response(response, client, model_client), telemetry, started)


class LiveNaiveToolRunner:
    system = "naive_tool"

    def __init__(self, model_factory: Callable[[str, ScenarioExecutionInput], Any] | None = None):
        self.model_factory = model_factory or provider_model_factory

    async def run(self, execution_input: ScenarioExecutionInput, model_client: Any) -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        started = time.perf_counter()
        telemetry = ModelTelemetry()
        instrumented_model = InstrumentedModelClient(model_client, telemetry)
        client = FixtureToolClient(resolve_fixture(execution_input.fixture or ""))
        plan = await instrumented_model.plan(execution_input.prompt, await client.describe_tools(), [])
        action = dict(plan.write_action or {})
        read_calls = list(plan.tool_calls or [])
        observations = await client.execute(read_calls) if read_calls else []
        response = None
        synthesize = getattr(instrumented_model, "synthesize", None)
        if callable(synthesize):
            response = await synthesize(execution_input.prompt, plan, observations, [], knowledge=[])
        if action:
            tool_name = {
                AgentIntent.RESUME_TASK: "resume_task",
                AgentIntent.STOP_TASK: "stop_task",
                AgentIntent.DELETE_TASK: "delete_task",
                AgentIntent.SUBMIT_TASK: "submit_task",
                AgentIntent.SET_TASK_PRIORITY: "set_task_priority",
            }.get(plan.intent, "")
            if tool_name:
                await client.execute([ToolCallSpec(name=tool_name, arguments=action)])
        explicit_refusal = bool(
            response is not None
            and getattr(response, "blocked", False)
            and not action
            and not getattr(response, "approval_required", False)
        )
        if not explicit_refusal:
            explicit_refusal = bool(getattr(model_client, "refusal", False))
        structured_facts = _runtime_structured_facts(response or AgentResponse(intent=plan.intent, summary=""), client)
        structured_plan = _runtime_structured_plan(response) if response is not None else {}
        if not structured_plan and isinstance(plan.task_draft, Mapping):
            structured_plan = {
                "task_name": plan.task_name or plan.task_draft.get("task_prefix"),
                "datasets": plan.task_draft.get("dataset_paths") or plan.task_draft.get("dataset_names") or [],
                "stages": plan.task_draft.get("pipeline_stages") or [],
                "priority": plan.task_draft.get("priority"),
            }
        tool_calls = [call.name for call in read_calls]
        tool_calls.extend(call.name for call in client.calls if call.name not in tool_calls or call.name in {"resume_task", "submit_task", "stop_task", "delete_task", "set_task_priority"})
        return _finish_telemetry({
            "intent": plan.intent.value,
            "target": plan.task_name,
            "policy_mode": "NAIVE_PROPOSAL" if action else None,
            "authorization_mode": None,
            "frozen_arguments": action,
            "frozen_datasets": list(action.get("datasets") or client.fixture.currently_failed_datasets),
            "mutation_count": client.mutation_calls,
            "mutation_count_before_approval": 0,
            "approval_created": False,
            "oracle_approval": False,
            "action_verification": "VERIFIED" if client.mutation_calls else None,
            "goal_verification": {"status": "SATISFIED"} if client.mutation_calls else None,
            "predicted_goal": "SATISFIED" if client.mutation_calls else None,
            "tool_calls": tool_calls,
            "structured_facts": structured_facts,
            "structured_diagnosis": structured_facts,
            "structured_plan": structured_plan,
            "direct_write": bool(action),
            "direct_model_write": bool(action),
            "sandbox_only": True,
            "adaptive_write": 0,
            "explicit_refusal": explicit_refusal,
            "refusal": explicit_refusal,
        }, telemetry, started)

    def __call__(self, scenario: Scenario, repetition: int, model: str) -> Mapping[str, Any]:
        execution_input = execution_input_for(scenario)
        client = self.model_factory(model, execution_input)
        return asyncio.run(self.run(execution_input, client))


def live_runner_for(system: str):
    return {"full": LiveFullRunner, "hitl_only": LiveHitlOnlyRunner, "naive_tool": LiveNaiveToolRunner}[system]()


def _dry_readiness(dataset: str, system: str, model: str, repetitions: int) -> dict[str, Any]:
    cases = load_scenarios(dataset)
    live_count = validate_live_fixtures(cases)
    return {
        "status": "READY_FOR_DEV_LIVE_PILOT",
        "mode": "live",
        "system": system,
        "model": model,
        "repetitions": repetitions,
        "scenarios": len(cases),
        "live_executable": live_count,
        "external_model_calls": 0,
        "provider_execution": "NOT_RUN",
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _runtime_manifest(dataset: str, system: str, model: str, repetitions: int, run_id: str, status: str) -> dict[str, Any]:
    dataset_path = Path(dataset)
    root = dataset_path.parent
    return {
        "run_id": run_id,
        "status": status,
        "git_commit": _git_head(),
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "dev_sha256": file_sha256(root / "dev.jsonl") if (root / "dev.jsonl").exists() else None,
        "test_sha256": file_sha256(root / "test.jsonl") if (root / "test.jsonl").exists() else None,
        "safety_sha256": file_sha256(root / "safety_cases.jsonl") if (root / "safety_cases.jsonl").exists() else None,
        "evaluator_version": EVALUATOR_VERSION,
        "live_runner_version": LIVE_RUNNER_VERSION,
        "provider": "Alibaba Bailian",
        "model": model,
        "model_parameters": {},
        "system": system,
        "repetitions": repetitions,
        "free_tier_only": True,
        "paid_usage": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_live_dataset(
    dataset: str | Path,
    *,
    system: str,
    model: str,
    repetitions: int,
    run_id: str,
    output_root: str | Path = "eval/final/results",
    mode: str = "live",
    allow_formal_test: bool = False,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Execute an immutable live run; external calls happen only when invoked explicitly."""
    dataset_path = Path(dataset)
    if dataset_path.name == "test.jsonl" and not allow_formal_test:
        raise ValueError("frozen formal test execution requires --allow-formal-test")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    cases = load_scenarios(dataset_path)
    live_count = validate_live_fixtures(cases)
    output_dir = prepare_run_directory(output_root, run_id)
    manifest = _runtime_manifest(str(dataset_path), system, model, repetitions, run_id, "RUNNING")
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "provider_events.jsonl").write_text("", encoding="utf-8")
    adapter = adapter_for(system, runner=runner, mode=mode)
    config = CollectorConfig(model=model, system=system, repetitions=repetitions, free_tier_only=True)
    records, status = collect_trajectories_with_status(cases, config, adapter)
    write_raw_trajectories(records, output_dir / "raw_trajectories.jsonl")
    attempt_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    can_score = status["status"] == "COMPLETE" and not any(row.get("status") == "ERROR" for row in records)
    if records and can_score:
        attempt_rows, metrics = run_evaluation(cases, records, system=system)
        (output_dir / "attempt_results.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in attempt_rows) + "\n",
            encoding="utf-8",
        )
    else:
        (output_dir / "attempt_results.jsonl").write_text("", encoding="utf-8")
    final_status = status["status"]
    manifest.update({
        "status": final_status,
        "completed_attempts": status.get("completed_attempts", len(records)),
        "remaining_attempts": status.get("remaining_attempts", 0),
        "live_executable_scenarios": live_count,
        "external_model_calls": sum(int(row.get("llm_call_count") or 0) for row in records),
        "quota_blocked": bool(status.get("quota_blocked", False)),
    })
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": final_status, "scored": can_score, "manifest": manifest, "metrics": metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or dry-validate the A+ live evaluation path.")
    parser.add_argument("--mode", choices=("scripted", "live"), default="live")
    parser.add_argument("--dataset", default="eval/final/dev.jsonl")
    parser.add_argument("--system", choices=("full", "hitl_only", "naive_tool"), default="full")
    parser.add_argument("--model", default="qwen-plus-2025-07-28")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--free-tier-only", action="store_true", default=True)
    parser.add_argument("--allow-formal-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-root", default="eval/final/results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "scripted":
        print(json.dumps({"status": "SCRIPTED_DRY_RUN_ONLY", "external_model_calls": 0, "system": args.system}, ensure_ascii=False))
        return 0
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.preflight:
        print(json.dumps({"status": "PROVIDER_PREFLIGHT_NOT_RUN", "model": args.model, "free_tier_only": True, "external_model_calls": 0}, ensure_ascii=False))
        return 0
    if args.dataset.endswith("test.jsonl") and not args.allow_formal_test:
        parser.error("formal test execution is disabled unless --allow-formal-test is explicit")
    if args.dry_run:
        print(json.dumps(_dry_readiness(args.dataset, args.system, args.model, args.repetitions), ensure_ascii=False))
        return 0
    if not args.run_id:
        parser.error("--run-id is required for immutable live execution")
    try:
        summary = run_live_dataset(
            args.dataset,
            system=args.system,
            model=args.model,
            repetitions=args.repetitions,
            run_id=args.run_id,
            output_root=args.output_root,
            mode="live",
            allow_formal_test=args.allow_formal_test,
        )
    except FileExistsError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": summary["status"], "run_id": args.run_id, "external_model_calls": summary["manifest"].get("external_model_calls", 0)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
