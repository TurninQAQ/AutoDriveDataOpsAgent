from __future__ import annotations

import csv
import json
from pathlib import Path

from .baselines.generic_hitl import GenericHITL
from .baselines.guarded_react import GuardedReAct
from .baselines.naive_react import NaiveReAct
from .harness import BenchmarkHarness
from .metrics import summarize
from .models import BenchmarkCase


def load_cases(case_dir: Path):
    rows=[]
    for path in sorted(case_dir.glob("*.jsonl")):
        rows += [BenchmarkCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


async def run_benchmark(case_dir: Path, out_dir: Path):
    """Run real deterministic baselines against isolated simulated platform state.

    GuardedReAct executes the production AgentRuntime/LangGraph/MCP path with a
    deterministic ScriptedProvider. No benchmark outcome is selected from a fault
    string; metrics are derived from actual tool trace, mutation records and final
    platform/FinalGuard state.
    """
    try:
        import mcp  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("real offline benchmark requires installed release-core mcp/langgraph dependencies") from exc

    cases=load_cases(case_dir); baselines=[NaiveReAct(),GenericHITL(),GuardedReAct()]
    outcomes=[]
    work_dir=out_dir/"benchmark_runtime"
    for baseline in baselines:
        for case in cases:
            harness=BenchmarkHarness(work_dir/baseline.name/case.case_id)
            outcomes.append(await baseline.run(case,harness))
    summary={baseline.name:summarize([r for r in outcomes if r.baseline==baseline.name]) for baseline in baselines}
    out_dir.mkdir(parents=True,exist_ok=True)
    payload={"benchmark_type":"deterministic_real_runtime_simulated_platform","case_count":len(cases),"summary":summary,"outcomes":[r.model_dump(mode="json") for r in outcomes]}
    (out_dir/"benchmark_results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out_dir/"benchmark_results.csv").open("w",newline="",encoding="utf-8") as h:
        fields=list(outcomes[0].model_dump().keys()) if outcomes else []
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for row in outcomes:
            data=row.model_dump(); data["mutation_targets"]=json.dumps(data["mutation_targets"],ensure_ascii=False); data["tool_trace"]=json.dumps(data["tool_trace"],ensure_ascii=False); w.writerow(data)
    lines=["# Offline Benchmark Summary","","Deterministic ScriptedProvider + real production Guarded AgentRuntime/LangGraph/MCP + isolated simulated platform.",""]
    for name,m in summary.items(): lines += [f"## {name}",*(f"- {k}: {v}" for k,v in m.items()),""]
    (out_dir/"benchmark_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return payload
