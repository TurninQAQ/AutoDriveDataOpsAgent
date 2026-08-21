from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .models import AgentResponse
from .runtime import build_agent_knowledge_service, build_default_agent, build_knowledge_service
from .settings import AgentSettings
from .tool_client import InMemoryMCPToolClient
from platform_core.settings import PlatformSettings
from platform_mcp.facade import build_default_facade
from platform_rag.evaluation import evaluate_retrieval
from platform_planning.service import TaskPlanningService
from platform_planning.evaluation import evaluate_task_planning
from platform_eval import evaluate_agent_suite
from platform_eval.aligned import evaluate_v11_suite
from platform_eval.frameworks import framework_status
from platform_eval.semantic import run_ragas_on_agent, run_deepeval_on_agent
from platform_observability import TraceStore
from platform_hardening import run_doctor, run_local_e2e


def _print_response(response: AgentResponse, as_json: bool = False) -> None:
    if as_json:
        print(response.model_dump_json(indent=2))
        return
    print(response.summary)
    if response.root_cause:
        print(f"\nRoot cause: {response.root_cause}")
    if response.evidence:
        print("\nEvidence:")
        for item in response.evidence:
            print(f"- {item}")
    if response.knowledge_sources:
        print("\nKnowledge sources:")
        for item in response.knowledge_sources:
            print(f"- {item}")
    if response.task_plan:
        plan = response.task_plan
        print("\nTask plan:")
        print(f"- valid={str(plan.get('valid', False)).lower()}")
        print(f"- unresolved={', '.join(plan.get('unresolved_fields') or []) or 'none'}")
        if plan.get("yaml_text"):
            print("\nGenerated YAML:\n")
            print(plan["yaml_text"].rstrip())
    if response.approval_required:
        print("\nApproval:")
        print(f"- approval_id={response.approval_id}")
        if response.pending_action:
            print(f"- tool={response.pending_action.get('tool_name')}")
            print(f"- risk={response.pending_action.get('risk_level')}")
            print(f"- impact={response.pending_action.get('impact_summary')}")
    if response.action_result:
        print("\nAction result:")
        print(json.dumps(response.action_result, ensure_ascii=False, indent=2, default=str))
    if response.recommended_next_actions:
        print("\nRecommended next actions:")
        for item in response.recommended_next_actions:
            print(f"- {item}")
    if response.errors:
        print("\nEvidence/retrieval errors:", file=sys.stderr)
        for item in response.errors:
            print(f"- {item}", file=sys.stderr)
    print(f"\nconfidence={response.confidence} intent={response.intent.value} blocked={str(response.blocked).lower()}")
    if response.trace_id:
        print(f"trace_id={response.trace_id}")


async def _ask(args) -> int:
    agent = build_default_agent()
    response = await agent.run(args.query, thread_id=args.thread_id)
    _print_response(response, args.json)
    return 0


async def _chat(args) -> int:
    agent = build_default_agent()
    print(f"DataOps guarded agent. thread_id={args.thread_id}. Write requests return an approval id. Type /exit to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in {"/exit", "/quit", "exit", "quit"}:
            return 0
        response = await agent.run(text, thread_id=args.thread_id)
        _print_response(response, False)
        print()


async def _tools(args) -> int:
    platform_settings = PlatformSettings.from_env()
    agent_settings = AgentSettings.from_env(platform_settings)
    client = InMemoryMCPToolClient(
        build_default_facade(
            platform_settings,
            knowledge_service=build_agent_knowledge_service(agent_settings),
        )
    )
    tools = await client.describe_tools()
    if args.json:
        print(json.dumps(tools, ensure_ascii=False, indent=2, default=str))
    else:
        for item in tools:
            print(f"{item.get('name')}: {item.get('description') or ''}")
    return 0


def _knowledge_service():
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    return build_knowledge_service(settings)


def _knowledge_status(args) -> int:
    status = _knowledge_service().status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"source_dir={status['source_dir']}")
        print(f"index_file={status['index_file']}")
        print(f"source_exists={str(status['source_exists']).lower()}")
        print(f"index_exists={str(status['index_exists']).lower()}")
        print(f"index_fresh={str(status['index_fresh']).lower()}")
        print(f"retrieval_mode={status.get('retrieval_mode')}")
        print(f"lexical_weight={status.get('lexical_weight')} vector_weight={status.get('vector_weight')}")
        embedding = status.get("embedding") or {}
        print(
            f"embedding_enabled={str(bool(embedding.get('enabled'))).lower()} "
            f"provider={embedding.get('provider')} model={embedding.get('model')} "
            f"dimension={embedding.get('dimension')} vectors={embedding.get('vector_count', 0)}"
        )
        stats = status.get("stats") or {}
        if stats:
            print(f"documents={stats.get('document_count')} chunks={stats.get('chunk_count')} built_at={stats.get('built_at')}")
    return 0


def _knowledge_build(args) -> int:
    stats = _knowledge_service().build(force=args.force, reset_embeddings=args.reset_embeddings)
    if args.json:
        print(stats.model_dump_json(indent=2))
    else:
        print(
            f"knowledge index ready: documents={stats.document_count} chunks={stats.chunk_count} "
            f"built_at={stats.built_at}"
        )
    return 0


def _knowledge_search(args) -> int:
    result = _knowledge_service().search(args.query, top_k=args.top_k)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    if not result.results:
        print("No knowledge chunk matched the query.")
        return 0
    for idx, item in enumerate(result.results, 1):
        citation = item.citation
        preview = " ".join(item.content.strip().split())[:260]
        print(f"{idx}. score={item.score:.3f} source={citation}")
        print(f"   {preview}")
    return 0


def _knowledge_eval(args) -> int:
    root = Path(__file__).resolve().parents[1]
    cases_path = Path(args.cases) if args.cases else root / "eval" / "rag_cases.json"
    result = evaluate_retrieval(_knowledge_service(), cases_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"RAG retrieval eval: cases={result['case_count']} hit@k={result['hit_at_k']:.3f} mrr={result['mrr']:.3f}")
        for row in result["cases"]:
            print(f"- {row['id']}: hit={str(row['hit']).lower()} rank={row['first_relevant_rank']}")
    return 0 if result["hit_at_k"] >= args.min_hit_rate else 1


def _plan_task(args) -> int:
    service = TaskPlanningService.from_env()
    result = service.plan(args.query)
    if args.output:
        try:
            written = service.write_yaml(result, args.output)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
    else:
        written = None
    if args.json:
        payload = result.model_dump(mode="json")
        if written:
            payload["output_path"] = str(written)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"valid={str(result.valid).lower()} priority={result.resolved_priority} source={result.priority_source or 'n/a'}")
        print(f"defaults_used={', '.join(result.defaults_used) or 'none'}")
        print(f"unresolved={', '.join(result.unresolved_fields) or 'none'}")
        if result.issues:
            print("Issues:")
            for issue in result.issues:
                print(f"- [{issue.severity}] {issue.code} {issue.path}: {issue.message}")
        print("\nGenerated YAML:\n")
        print(result.yaml_text.rstrip())
        if written:
            print(f"\nwritten={written}")
        print("\nNo task was submitted.")
    return 0 if result.valid else 1


def _plan_task_eval(args) -> int:
    root = Path(__file__).resolve().parents[1]
    cases_path = Path(args.cases) if args.cases else root / "eval" / "task_planning_cases.json"
    result = evaluate_task_planning(TaskPlanningService.from_env(), cases_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"Task planning eval: cases={result['case_count']} passed={result['passed']} "
            f"accuracy={result['case_accuracy']:.3f}"
        )
        for row in result["cases"]:
            print(f"- {row['id']}: ok={str(row['ok']).lower()}")
    return 0 if result["case_accuracy"] >= args.min_accuracy else 1


def _print_approval(item, as_json: bool = False) -> None:
    if as_json:
        print(item.model_dump_json(indent=2))
        return
    print(f"approval_id={item.approval_id} status={item.status} risk={item.risk_level} tool={item.tool_name}")
    print(f"impact={item.impact_summary}")
    for detail in item.impact_details:
        print(f"- {detail}")
    if item.error:
        print(f"error={item.error}")
    if item.execution_result is not None:
        print("execution_result=")
        print(json.dumps(item.execution_result, ensure_ascii=False, indent=2, default=str))
    if item.verification_result is not None:
        verification = item.verification_result
        print(f"verification_status={verification.get('status')} attempts={verification.get('attempts')}")
        for check in verification.get("checks") or []:
            print(f"- {check.get('name')} passed={str(bool(check.get('passed'))).lower()} expected={check.get('expected')} actual={check.get('actual')}")
    if item.goal_verification_result is not None:
        goal_verification = item.goal_verification_result
        print(f"goal_verification_status={goal_verification.get('status')} attempts={goal_verification.get('attempts')}")
        for check in goal_verification.get("checks") or []:
            print(f"- goal:{check.get('name')} passed={str(bool(check.get('passed'))).lower()} expected={check.get('expected')} actual={check.get('actual')}")


async def _approve(args) -> int:
    agent = build_default_agent()
    item = await agent.approve(args.approval_id)
    _print_approval(item, args.json)
    return 0 if item.status == "executed" else 1


def _reject(args) -> int:
    agent = build_default_agent()
    item = agent.reject(args.approval_id, args.reason)
    _print_approval(item, args.json)
    return 0


def _approvals(args) -> int:
    agent = build_default_agent()
    items = agent.approvals(args.status)
    if args.json:
        print(json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2, default=str))
    else:
        if not items:
            print("No approvals found.")
        for item in items:
            print(f"{item.approval_id} status={item.status} risk={item.risk_level} tool={item.tool_name} impact={item.impact_summary}")
    return 0



def _trace_store() -> TraceStore:
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    return TraceStore(settings.trace_dir, settings.audit_file)


def _traces(args) -> int:
    store = _trace_store()
    items = store.summaries(limit=args.limit)
    if args.json:
        print(json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2, default=str))
    else:
        if not items:
            print("No traces found.")
        for item in items:
            parent = f" parent={item.parent_trace_id}" if item.parent_trace_id else ""
            print(
                f"{item.trace_id} kind={item.kind} status={item.status} latency_ms={item.latency_ms:.1f} "
                f"intent={item.intent or '-'} errors={item.error_count}{parent}"
            )
            if item.user_request:
                print(f"  request={item.user_request[:180]}")
    return 0


def _trace_show(args) -> int:
    store = _trace_store()
    events = store.load_events(args.trace_id)
    if not events:
        print(f"Trace not found: {args.trace_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([item.model_dump(mode="json") for item in events], ensure_ascii=False, indent=2, default=str))
        return 0
    for item in events:
        dur = f" duration_ms={item.duration_ms:.1f}" if item.duration_ms is not None else ""
        print(f"{item.timestamp:.3f} stage={item.stage} name={item.name} status={item.status}{dur}")
        if item.data:
            print("  " + json.dumps(item.data, ensure_ascii=False, default=str)[:1000])
    return 0


def _eval_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _eval_aligned(args) -> int:
    root = _eval_root()
    base = Path(args.eval_dir) if args.eval_dir else root / "eval" / "v1_1"
    result = evaluate_v11_suite(
        knowledge_service=_knowledge_service(),
        rag_cases=base / "rag_retrieval.jsonl",
        tool_cases=base / "agent_tool_cases.jsonl",
        task_cases=base / "agent_task_cases.jsonl",
        security_cases=base / "security" / "curated_attacks.jsonl",
        planning_cases=root / "eval" / "task_planning_cases.json",
        planning_service=TaskPlanningService.from_env(),
    )
    thresholds_path = Path(args.thresholds) if args.thresholds else base / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    gates = result["gates"]
    limits = {
        "rag_context_recall_min": args.min_context_recall if args.min_context_recall is not None else float(thresholds["rag_context_recall_min"]),
        "rag_context_precision_min": args.min_context_precision if args.min_context_precision is not None else float(thresholds["rag_context_precision_min"]),
        "tool_f1_min": args.min_tool_f1 if args.min_tool_f1 is not None else float(thresholds["tool_f1_min"]),
        "argument_accuracy_min": args.min_argument_accuracy if args.min_argument_accuracy is not None else float(thresholds["argument_accuracy_min"]),
        "hard_task_success_rate_min": args.min_hard_task_success if args.min_hard_task_success is not None else float(thresholds["hard_task_success_rate_min"]),
        "task_planning_accuracy_min": args.min_planning_accuracy if args.min_planning_accuracy is not None else float(thresholds["task_planning_accuracy_min"]),
        "security_attack_success_rate_max": args.max_attack_success_rate if args.max_attack_success_rate is not None else float(thresholds["security_attack_success_rate_max"]),
    }
    ok = (
        gates["rag_context_recall"] >= limits["rag_context_recall_min"]
        and gates["rag_context_precision"] >= limits["rag_context_precision_min"]
        and gates["tool_f1"] >= limits["tool_f1_min"]
        and gates["argument_accuracy"] >= limits["argument_accuracy_min"]
        and gates["hard_task_success_rate"] >= limits["hard_task_success_rate_min"]
        and (gates.get("task_planning_accuracy") is None or gates["task_planning_accuracy"] >= limits["task_planning_accuracy_min"])
        and gates["security_attack_success_rate"] <= limits["security_attack_success_rate_max"]
    )
    result["thresholds"] = limits
    result["thresholds_source"] = str(thresholds_path)
    result["passed"] = ok
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print("V1.1 aligned evaluation:")
        print(f"- rag_context_recall={gates['rag_context_recall']:.3f} min={limits['rag_context_recall_min']:.3f}")
        print(f"- rag_context_precision={gates['rag_context_precision']:.3f} min={limits['rag_context_precision_min']:.3f}")
        print(f"- tool_f1={gates['tool_f1']:.3f} min={limits['tool_f1_min']:.3f}")
        print(f"- argument_accuracy={gates['argument_accuracy']:.3f} min={limits['argument_accuracy_min']:.3f}")
        print(f"- hard_task_success_rate={gates['hard_task_success_rate']:.3f} min={limits['hard_task_success_rate_min']:.3f}")
        if gates.get("task_planning_accuracy") is not None:
            print(f"- task_planning_accuracy={gates['task_planning_accuracy']:.3f} min={limits['task_planning_accuracy_min']:.3f}")
        print(f"- security_attack_success_rate={gates['security_attack_success_rate']:.3f} max={limits['security_attack_success_rate_max']:.3f}")
        print(f"- passed={str(ok).lower()}")
    return 0 if ok else 1


def _eval_frameworks(args) -> int:
    status = framework_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    else:
        for name, item in status.items():
            print(
                f"{name}: available={str(item['available']).lower()} "
                f"recommended={item['recommended_version']} purpose={item['purpose']}"
            )
    return 0


def _eval_ragas(args) -> int:
    root = _eval_root()
    cases = Path(args.cases) if args.cases else root / "eval" / "v1_1" / "rag_generation_cases.jsonl"
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    result = run_ragas_on_agent(_knowledge_service(), cases, settings)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Ragas semantic evaluation: cases={result.get('case_count', 0)}")
        for key, value in (result.get("metrics") or {}).items():
            print(f"- {key}={float(value):.3f}")
    return 0


def _eval_deepeval(args) -> int:
    root = _eval_root()
    cases = Path(args.cases) if args.cases else root / "eval" / "v1_1" / "agent_tool_cases.jsonl"
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    result = run_deepeval_on_agent(cases, settings)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result.get("status") == "COLLECTION_INVALID":
            return 2
    else:
        if result.get("status") == "COLLECTION_INVALID":
            print("DeepEval tool evaluation: status=COLLECTION_INVALID")
            for item in result.get("invalid_cases") or []:
                print(
                    f"- case={item.get('case_id')} required_tools={item.get('required_tools')} "
                    f"actual_tools={item.get('actual_tools')} reason={item.get('reason')}"
                )
            return 2
        print(f"DeepEval tool evaluation: cases={result.get('case_count', 0)}")
        print(f"- tool_correctness={float(result.get('tool_correctness', 0.0)):.3f}")
        print(f"- argument_correctness={float(result.get('argument_correctness', 0.0)):.3f}")
        print(f"- task_completion_note={result.get('task_completion_note', '')}")
    return 0


def _eval_promptfoo(args) -> int:
    root = _eval_root()
    security_dir = root / "eval" / "v1_1" / "security"
    config = Path(args.config) if args.config else security_dir / "promptfooconfig.yaml"
    binary = shutil.which("promptfoo")
    if binary:
        cmd = [binary]
    elif args.allow_npx and shutil.which("npx"):
        cmd = [shutil.which("npx") or "npx", "promptfoo@latest"]
    else:
        raise RuntimeError(
            "Promptfoo CLI is not installed. Install promptfoo, or pass --allow-npx to use `npx promptfoo@latest`."
        )
    if args.redteam:
        cmd += ["redteam", "run", "-c", str(config)]
    else:
        cmd += ["eval", "-c", str(config), "--no-progress-bar", "--no-write"]
    if args.output:
        cmd += ["--output", str(args.output)]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(root))
    result = subprocess.run(cmd, cwd=security_dir, env=env)
    return int(result.returncode)


def _agent_eval(args) -> int:
    root = Path(__file__).resolve().parents[1]
    cases = Path(args.cases) if args.cases else root / "eval" / "agent_cases.json"
    planning_cases = Path(args.planning_cases) if args.planning_cases else root / "eval" / "task_planning_cases.json"
    result = evaluate_agent_suite(cases, planning_cases, TaskPlanningService.from_env())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print("Agent evaluation:")
        for key in (
            "intent_accuracy", "tool_selection_accuracy", "diagnosis_accuracy",
            "unsafe_action_rate", "task_planning_accuracy", "verification_accuracy", "overall_score",
        ):
            print(f"- {key}={result[key]:.3f}")
    thresholds_ok = (
        result["intent_accuracy"] >= args.min_accuracy
        and result["tool_selection_accuracy"] >= args.min_accuracy
        and result["diagnosis_accuracy"] >= args.min_accuracy
        and result["task_planning_accuracy"] >= args.min_accuracy
        and result["verification_accuracy"] >= args.min_accuracy
        and result["unsafe_action_rate"] <= args.max_unsafe_rate
    )
    return 0 if thresholds_ok else 1


def _doctor(args) -> int:
    report = run_doctor()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(f"ready_dependency_light={str(report.ready_dependency_light).lower()}")
        print(f"ready_full_runtime={str(report.ready_full_runtime).lower()}")
        for item in report.checks:
            print(f"- [{item.status}] {item.name}: {item.detail}")
    return 0 if (report.ready_full_runtime if args.strict else report.ready_dependency_light) else 1


def _e2e(args) -> int:
    result = run_local_e2e(args.root or None)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"local_e2e_ok={str(result.ok).lower()} traces={result.trace_count} audits={result.audit_count}")
        for item in result.steps:
            print(f"- [{'PASS' if item.ok else 'FAIL'}] {item.name}: {item.detail}")
        if args.root:
            print(f"artifacts_root={result.artifacts_root}")
    return 0 if result.ok else 1


def _observability_maintenance(args) -> int:
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    result = _trace_store().maintenance(
        retention_days=settings.trace_retention_days,
        max_trace_files=settings.trace_max_files,
        audit_max_bytes=settings.audit_max_bytes,
        audit_backup_count=settings.audit_backup_count,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataops-agent",
        description="V1.1 read-only Agent planning surface with guarded HITL writes, verification, RAG, observability and aligned evaluation.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Run one Agent request; writes create a pending HITL approval")
    ask.add_argument("query")
    ask.add_argument("--thread-id", default="default")
    ask.add_argument("--json", action="store_true")

    chat = sub.add_parser("chat", help="Start a local terminal conversation")
    chat.add_argument("--thread-id", default="default")

    plan_task = sub.add_parser("plan-task", help="Generate and validate a local task YAML without submitting it")
    plan_task.add_argument("query")
    plan_task.add_argument("--output", default="")
    plan_task.add_argument("--json", action="store_true")

    plan_eval = sub.add_parser("plan-task-eval", help="Run deterministic natural-language TaskSpec evaluation cases")
    plan_eval.add_argument("--cases", default="")
    plan_eval.add_argument("--min-accuracy", type=float, default=1.0)
    plan_eval.add_argument("--json", action="store_true")

    tools = sub.add_parser("tools", help="List the read-side MCP tools visible to the planning model")
    tools.add_argument("--json", action="store_true")

    approve = sub.add_parser("approve", help="Approve and execute one frozen pending write action")
    approve.add_argument("approval_id")
    approve.add_argument("--json", action="store_true")

    reject = sub.add_parser("reject", help="Reject one pending write action")
    reject.add_argument("approval_id")
    reject.add_argument("--reason", default="Rejected by user")
    reject.add_argument("--json", action="store_true")

    approvals = sub.add_parser("approvals", help="List persisted HITL approvals")
    approvals.add_argument("--status", default="pending", choices=["pending", "executing", "rejected", "executed", "failed", "verification_failed", "expired", ""])
    approvals.add_argument("--json", action="store_true")

    traces = sub.add_parser("traces", help="List persisted Agent/Audit trace summaries")
    traces.add_argument("--limit", type=int, default=50)
    traces.add_argument("--json", action="store_true")

    trace = sub.add_parser("trace", help="Inspect one persisted trace")
    trace.add_argument("trace_id")
    trace.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="Check dependency-light and full-runtime readiness")
    doctor.add_argument("--strict", action="store_true", help="Fail unless Airflow/MCP/LangGraph/Docker/GPU full runtime is ready")
    doctor.add_argument("--json", action="store_true")

    e2e = sub.add_parser("e2e", help="Run dependency-light V1.0 end-to-end hardening scenario")
    e2e.add_argument("--root", default="", help="Persist E2E artifacts in this directory instead of a temporary directory")
    e2e.add_argument("--json", action="store_true")

    obs = sub.add_parser("observability-maintenance", help="Prune old traces and rotate the audit JSONL file")
    obs.add_argument("--json", action="store_true")

    aligned_eval = sub.add_parser("eval-aligned", help="Run V1.1 dependency-light RAG/Agent/security evaluation gates")
    aligned_eval.add_argument("--eval-dir", default="", help="Override eval/v1_1 fixture directory")
    aligned_eval.add_argument("--thresholds", default="", help="Override eval/v1_1/thresholds.json")
    aligned_eval.add_argument("--min-context-recall", type=float, default=None)
    aligned_eval.add_argument("--min-context-precision", type=float, default=None)
    aligned_eval.add_argument("--min-tool-f1", type=float, default=None)
    aligned_eval.add_argument("--min-argument-accuracy", type=float, default=None)
    aligned_eval.add_argument("--min-hard-task-success", type=float, default=None)
    aligned_eval.add_argument("--min-planning-accuracy", type=float, default=None)
    aligned_eval.add_argument("--max-attack-success-rate", type=float, default=None)
    aligned_eval.add_argument("--json", action="store_true")

    eval_frameworks = sub.add_parser("eval-frameworks", help="Show optional Ragas/DeepEval/Promptfoo availability")
    eval_frameworks.add_argument("--json", action="store_true")

    eval_ragas = sub.add_parser("eval-ragas", help="Run optional native Ragas semantic judge metrics on real model output")
    eval_ragas.add_argument("--cases", default="")
    eval_ragas.add_argument("--json", action="store_true")

    eval_deepeval = sub.add_parser("eval-deepeval", help="Run optional native DeepEval tool/argument metrics on real model plans")
    eval_deepeval.add_argument("--cases", default="")
    eval_deepeval.add_argument("--json", action="store_true")

    eval_promptfoo = sub.add_parser("eval-promptfoo", help="Run Promptfoo curated safety eval or dynamic red-team config")
    eval_promptfoo.add_argument("--config", default="")
    eval_promptfoo.add_argument("--redteam", action="store_true", help="Run `promptfoo redteam run` instead of curated eval")
    eval_promptfoo.add_argument("--allow-npx", action="store_true", help="Allow npx promptfoo@latest when promptfoo binary is absent")
    eval_promptfoo.add_argument("--output", default="")

    agent_eval = sub.add_parser("eval", help="Run legacy deterministic V0.9 Agent regression suite")
    agent_eval.add_argument("--cases", default="")
    agent_eval.add_argument("--planning-cases", default="")
    agent_eval.add_argument("--min-accuracy", type=float, default=1.0)
    agent_eval.add_argument("--max-unsafe-rate", type=float, default=0.0)
    agent_eval.add_argument("--json", action="store_true")

    knowledge = sub.add_parser("knowledge", help="Build, inspect or search the static platform knowledge index")
    ksub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    status = ksub.add_parser("status", help="Show source/index status")
    status.add_argument("--json", action="store_true")
    build = ksub.add_parser("build", help="Build or refresh the knowledge index")
    build.add_argument("--force", action="store_true")
    build.add_argument("--reset-embeddings", action="store_true", help="Discard the dense sidecar and rebuild all vectors")
    build.add_argument("--json", action="store_true")
    search = ksub.add_parser("search", help="Search platform knowledge without running the Agent")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--json", action="store_true")
    evaluate = ksub.add_parser("eval", help="Run deterministic retrieval evaluation cases")
    evaluate.add_argument("--cases", default="")
    evaluate.add_argument("--min-hit-rate", type=float, default=1.0)
    evaluate.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "plan-task":
            return _plan_task(args)
        if args.command == "plan-task-eval":
            return _plan_task_eval(args)
        if args.command == "tools":
            return asyncio.run(_tools(args))
        if args.command == "approve":
            return asyncio.run(_approve(args))
        if args.command == "reject":
            return _reject(args)
        if args.command == "approvals":
            return _approvals(args)
        if args.command == "traces":
            return _traces(args)
        if args.command == "trace":
            return _trace_show(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "e2e":
            return _e2e(args)
        if args.command == "observability-maintenance":
            return _observability_maintenance(args)
        if args.command == "eval-aligned":
            return _eval_aligned(args)
        if args.command == "eval-frameworks":
            return _eval_frameworks(args)
        if args.command == "eval-ragas":
            return _eval_ragas(args)
        if args.command == "eval-deepeval":
            return _eval_deepeval(args)
        if args.command == "eval-promptfoo":
            return _eval_promptfoo(args)
        if args.command == "eval":
            return _agent_eval(args)
        if args.command == "knowledge":
            if args.knowledge_command == "status":
                return _knowledge_status(args)
            if args.knowledge_command == "build":
                return _knowledge_build(args)
            if args.knowledge_command == "search":
                return _knowledge_search(args)
            if args.knowledge_command == "eval":
                return _knowledge_eval(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 2
