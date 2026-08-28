from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from deploy_ci_cloud_agentv3 import __version__


def _interrupt_payload(state):
    items = state.get("__interrupt__") if isinstance(state, dict) else None
    if not items: return None
    first=items[0]
    return getattr(first,"value",first)


async def _run_query(query: str) -> int:
    from deploy_ci_cloud_agentv3.config import Settings
    from deploy_ci_cloud_agentv3.persistence.checkpoint import CheckpointerFactory
    from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
    from deploy_ci_cloud_agentv3.providers.qwen import QwenProvider
    settings=Settings.from_env(); settings.ensure_dirs()
    async with CheckpointerFactory.open(settings.checkpoint_backend,path=settings.checkpoint_path) as saver:
        runtime=AgentRuntime.local(QwenProvider(),checkpointer=saver,audit_path=str(settings.db_path))
        thread_id=f"cli_{uuid.uuid4().hex}"
        state=await runtime.start(thread_id,query)
        while True:
            interrupt=_interrupt_payload(state)
            if not interrupt:
                print(json.dumps(state.get("final_response") or state,ensure_ascii=False,indent=2,default=str)); return 0
            print(json.dumps(interrupt,ensure_ascii=False,indent=2,default=str))
            raw=input("review [approve/reject/edit JSON]: ").strip()
            if raw.startswith("{"): decision=json.loads(raw)
            elif raw=="approve": decision={"decision":"approve","fingerprint":interrupt.get("fingerprint")}
            else: decision={"decision":raw or "reject"}
            state=await runtime.review(thread_id,decision)


async def _rag_build(deterministic_test: bool=False) -> int:
    from deploy_ci_cloud_agentv3.config import Settings
    from deploy_ci_cloud_agentv3.rag.factory import build_embedding_provider
    from deploy_ci_cloud_agentv3.rag.index import DenseIndex
    settings=Settings.from_env(); settings.ensure_dirs()
    provider=build_embedding_provider(settings,test_deterministic=deterministic_test)
    if provider is None: raise RuntimeError("RAG_DENSE_PROVIDER=disabled; configure gemini or use --deterministic-test for test-only index")
    source=Path(__file__).resolve().parent/"platform_backend"/"knowledge"
    manifest=await DenseIndex(settings.state_dir/"knowledge_index").build(source,provider)
    print(json.dumps(manifest,indent=2,ensure_ascii=False)); return 0


async def _rag_eval(deterministic_test: bool=False) -> int:
    from deploy_ci_cloud_agentv3.config import Settings
    from deploy_ci_cloud_agentv3.rag.factory import build_rag_service
    from deploy_ci_cloud_agentv3.rag.index import DenseIndex
    from deploy_ci_cloud_agentv3.rag.evaluation import load_cases,evaluate,write_artifacts
    settings=Settings.from_env(); settings.ensure_dirs()
    service=build_rag_service(settings=settings,test_deterministic=deterministic_test)
    if deterministic_test and service.embedding_provider and not service.index.is_fresh(service.source_dir,model=service.embedding_provider.model_name,dimension=service.embedding_provider.dimension):
        await service.index.build(service.source_dir,service.embedding_provider)
    cases=load_cases(Path(__file__).resolve().parent/"evaluation"/"rag_cases.jsonl")
    report=await evaluate(service,cases)
    report["embedding_evaluation"]="deterministic_test_only" if deterministic_test else ("real_external" if service.effective_mode in {"dense","hybrid"} else "not_run_external")
    write_artifacts(report,Path("artifacts")); print(json.dumps({k:v for k,v in report.items() if k!="cases"},indent=2)); return 0


async def _benchmark() -> int:
    from deploy_ci_cloud_agentv3.evaluation.runner import run_benchmark
    root=Path(__file__).resolve().parent/"evaluation"/"cases"
    report=await run_benchmark(root,Path("artifacts")); print(json.dumps(report["summary"],indent=2)); return 0


def _serve_api() -> int:
    import uvicorn
    from deploy_ci_cloud_agentv3.config import Settings
    settings=Settings.from_env()
    uvicorn.run("deploy_ci_cloud_agentv3.api.app:app",host=settings.api_host,port=settings.api_port,factory=False)
    return 0


def _serve_mcp() -> int:
    from deploy_ci_cloud_agentv3.mcp.server import main
    main(); return 0


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="AutoDriveDataOpsAgent V3.9")
    p.add_argument("--version",action="version",version=f"%(prog)s {__version__}")
    sub=p.add_subparsers(dest="command")
    run=sub.add_parser("run"); run.add_argument("query",nargs="+")
    sub.add_parser("serve"); sub.add_parser("mcp-serve"); sub.add_parser("health"); sub.add_parser("ready")
    rag=sub.add_parser("rag"); rsub=rag.add_subparsers(dest="rag_command",required=True)
    b=rsub.add_parser("build-index"); b.add_argument("--deterministic-test",action="store_true")
    e=rsub.add_parser("eval"); e.add_argument("--deterministic-test",action="store_true")
    sub.add_parser("benchmark")
    return p


def main() -> None:
    argv=sys.argv[1:]
    known={"run","serve","mcp-serve","health","ready","rag","benchmark","-h","--help","--version"}
    if argv and argv[0] not in known:
        argv=["run",*argv]
    args=build_parser().parse_args(argv)
    if args.command=="run": raise SystemExit(asyncio.run(_run_query(" ".join(args.query))))
    if args.command=="serve": raise SystemExit(_serve_api())
    if args.command=="mcp-serve": raise SystemExit(_serve_mcp())
    if args.command=="health":
        print(json.dumps({"status":"ok","version":__version__})); raise SystemExit(0)
    if args.command=="ready":
        from deploy_ci_cloud_agentv3.config import Settings
        from deploy_ci_cloud_agentv3.persistence.database import initialize
        import os
        settings=Settings.from_env(); settings.ensure_dirs(); initialize(settings.db_path)
        problems=[]
        for module in ("mcp","langgraph"):
            try: __import__(module)
            except Exception: problems.append(f"missing {module}")
        if not (os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
            problems.append("missing Qwen API key")
        print(json.dumps({"status":"ready" if not problems else "not_ready","problems":problems,"version":__version__}))
        raise SystemExit(0 if not problems else 1)
    if args.command=="rag" and args.rag_command=="build-index": raise SystemExit(asyncio.run(_rag_build(args.deterministic_test)))
    if args.command=="rag" and args.rag_command=="eval": raise SystemExit(asyncio.run(_rag_eval(args.deterministic_test)))
    if args.command=="benchmark": raise SystemExit(asyncio.run(_benchmark()))
    build_parser().print_help()

if __name__=="__main__": main()
