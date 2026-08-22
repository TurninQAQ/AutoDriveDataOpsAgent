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
import tempfile
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
from .collector import QuotaBlockedError, translate_provider_exception


LIVE_RUNNER_VERSION = "a-plus-final-live-runner-v1"


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

    async def describe_tools(self) -> list[dict[str, Any]]:
        from platform_agent.tool_catalog import build_read_only_tool_catalog

        return build_read_only_tool_catalog(knowledge_enabled=True) + [
            {"name": name, "description": name, "input_schema": {}}
            for name in ("get_write_precondition", "get_action_verification_snapshot", "resume_task", "submit_task", "stop_task", "delete_task", "set_task_priority")
        ]

    def _runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for dataset, state in self.fixture.latest_dataset_states.items():
            runs.append({"run_id": f"old-{dataset}", "dataset_name": dataset, "state": state})
        return runs + list(self._new_runs)

    def _snapshot(self, task_name: str) -> dict[str, Any]:
        observed_task = "other_task" if self.fixture.provenance_conflict else task_name
        return {
            "task_name": observed_task,
            "task_exists": self.fixture.task_exists,
            "config_file_exists": self.fixture.config_exists,
            "dag_file_exists": self.fixture.dag_exists,
            "airflow_dag_exists": self.fixture.dag_exists,
            "available_datasets": list(self.fixture.available_datasets),
            "task_exclusive": self.fixture.task_exclusive,
            "queue": {
                "location": "queued" if self.fixture.task_exclusive else "not_found",
                "position": 0 if self.fixture.task_exclusive else -1,
                "entry": {"task_name": task_name} if self.fixture.task_exclusive else None,
            },
            "active_task_name": self.fixture.active_task_name or "",
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
                    "task_exists": self.fixture.task_exists,
                    "config_file_exists": self.fixture.config_exists,
                    "dag_file_exists": self.fixture.dag_exists,
                    "datasets": list(self.fixture.available_datasets),
                    "state": next(iter(self.fixture.latest_dataset_states.values()), "unknown"),
                    "task_state": next(iter(self.fixture.latest_dataset_states.values()), "unknown"),
                }
                result.append(ToolObservation(tool_name=name, arguments=args, ok=True, data=data))
            elif name == "get_queue_state":
                active = None
                if self.fixture.active_task_name:
                    active = {"task_name": self.fixture.active_task_name, "priority": 1}
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
                        "task_exists": self.fixture.task_exists,
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


def _build_agent(execution_input: ScenarioExecutionInput, model: FakeLLMClient, *, autonomy_enabled: bool, root: Path):
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


def _raw_response(response: AgentResponse, client: FixtureToolClient, model: FakeLLMClient, *, approval_count: int = 0, mutation_count_before_approval: int = 0) -> dict[str, Any]:
    pending = response.pending_action or {}
    arguments = pending.get("arguments") if isinstance(pending, Mapping) else {}
    arguments = arguments if isinstance(arguments, Mapping) else {}
    structured_facts = dict(model.structured_facts)
    if not structured_facts:
        for observation in reversed(client.observations):
            if observation.tool_name in {"diagnose_task", "get_task_detail", "get_gpu_pool"} and isinstance(observation.data, Mapping):
                structured_facts = {str(key): value for key, value in observation.data.items() if key not in {"task_name", "task_exists", "config_file_exists", "dag_file_exists", "datasets"}}
                if structured_facts:
                    break
    structured_plan = dict(model.structured_plan or {})
    if not structured_plan and isinstance(response.task_plan, Mapping):
        task_spec = response.task_plan.get("task_spec") or {}
        if isinstance(task_spec, Mapping):
            structured_plan = {
                "task_name": task_spec.get("task_prefix"),
                "datasets": [item.get("dataset_path") for item in task_spec.get("datasets") or [] if isinstance(item, Mapping)],
                "stages": task_spec.get("pipeline_stages") or [],
                "priority": response.task_plan.get("resolved_priority", task_spec.get("priority")),
            }
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
        "refusal": model.refusal,
    }


class LiveFullRunner:
    system = "full"

    def __init__(self, model_factory: Callable[[str, ScenarioExecutionInput], Any] | None = None):
        self.model_factory = model_factory or provider_model_factory

    async def run(self, execution_input: ScenarioExecutionInput, model_client: FakeLLMClient, *, root: Path | None = None) -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        work_root = root or Path(tempfile.mkdtemp(prefix="a-plus-live-full-"))
        agent, client, _store = _build_agent(execution_input, model_client, autonomy_enabled=True, root=work_root)
        try:
            response = await agent.run(execution_input.prompt, f"live-{execution_input.case_id}")
        except Exception as exc:
            translated = translate_provider_exception(exc, model="live-model")
            if isinstance(translated, QuotaBlockedError):
                raise translated
            raise
        return _raw_response(response, client, model_client)

    def __call__(self, scenario: Scenario, repetition: int, model: str) -> Mapping[str, Any]:
        if self.model_factory is None:
            raise RuntimeError("LiveFullRunner requires an explicit provider/model factory")
        execution_input = execution_input_for(scenario)
        client = self.model_factory(model, execution_input)
        return asyncio.run(self.run(execution_input, client))


class LiveHitlOnlyRunner(LiveFullRunner):
    system = "hitl_only"

    async def run(self, execution_input: ScenarioExecutionInput, model_client: FakeLLMClient, *, root: Path | None = None) -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        work_root = root or Path(tempfile.mkdtemp(prefix="a-plus-live-hitl-"))
        agent, client, _store = _build_agent(execution_input, model_client, autonomy_enabled=False, root=work_root)
        try:
            response = await agent.run(execution_input.prompt, f"live-{execution_input.case_id}")
        except Exception as exc:
            translated = translate_provider_exception(exc, model="live-model")
            if isinstance(translated, QuotaBlockedError):
                raise translated
            raise
        if response.approval_required and response.authorization_mode == "hitl" and (response.policy_decision or {}).get("mode") == "HITL":
            before = client.mutation_calls
            item = await agent.approve(response.approval_id)
            response.action_verification = item.verification_result
            response.goal_verification_result = item.goal_verification_result
            response.pending_action = item.model_dump(mode="json")
            response.action_result = item.execution_result
            return _raw_response(response, client, model_client, approval_count=1, mutation_count_before_approval=before)
        return _raw_response(response, client, model_client)


class LiveNaiveToolRunner:
    system = "naive_tool"

    def __init__(self, model_factory: Callable[[str, ScenarioExecutionInput], Any] | None = None):
        self.model_factory = model_factory or provider_model_factory

    async def run(self, execution_input: ScenarioExecutionInput, model_client: FakeLLMClient) -> dict[str, Any]:
        assert_ground_truth_isolated(execution_input)
        client = FixtureToolClient(resolve_fixture(execution_input.fixture or ""))
        plan = await model_client.plan(execution_input.prompt, await client.describe_tools(), [])
        action = dict(plan.write_action or {})
        if action:
            tool_name = {AgentIntent.RESUME_TASK: "resume_task", AgentIntent.STOP_TASK: "stop_task", AgentIntent.DELETE_TASK: "delete_task", AgentIntent.SUBMIT_TASK: "submit_task"}.get(plan.intent, "")
            if tool_name:
                await client.execute([ToolCallSpec(name=tool_name, arguments=action)])
        return {
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
            "tool_calls": [call.name for call in client.calls],
            "direct_write": bool(action),
            "direct_model_write": bool(action),
            "sandbox_only": True,
            "adaptive_write": 0,
            "refusal": model_client.refusal,
        }

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or dry-validate the A+ live evaluation path.")
    parser.add_argument("--mode", choices=("scripted", "live"), required=True)
    parser.add_argument("--dataset", default="eval/final/dev.jsonl")
    parser.add_argument("--system", choices=("full", "hitl_only", "naive_tool"), default="full")
    parser.add_argument("--model", default="qwen-plus-2025-07-28")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--free-tier-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "live" and not args.dry_run:
        parser.error("provider execution is disabled in the readiness gate; use --dry-run")
    if args.mode == "scripted":
        print(json.dumps({"status": "SCRIPTED_DRY_RUN_ONLY", "external_model_calls": 0, "system": args.system}, ensure_ascii=False))
        return 0
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    print(json.dumps(_dry_readiness(args.dataset, args.system, args.model, args.repetitions), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
