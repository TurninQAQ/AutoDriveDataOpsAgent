from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .service import RAGService


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def evaluate(service: RAGService, cases: list[dict[str, Any]], *, top_k: int = 5) -> dict[str, Any]:
    rows=[]
    reciprocal=[]
    hits={1:0,3:0,5:0}
    for case in cases:
        result=await service.search(case["query"], top_k=max(top_k,5))
        relevant=set(case.get("relevant_sources") or [])
        ranked=[str(item["source"]) for item in result["results"]]
        first=None
        for rank,source in enumerate(ranked,1):
            if any(source == rel or source.endswith(rel) for rel in relevant):
                first=rank; break
        reciprocal.append(0.0 if first is None else 1.0/first)
        for k in hits:
            if first is not None and first <= k: hits[k]+=1
        rows.append({"case_id":case["case_id"],"query":case["query"],"mode":result["mode"],"first_relevant_rank":first,"top_sources":ranked[:5]})
    n=max(1,len(cases))
    return {"mode": service.effective_mode, "case_count":len(cases), "recall_at_1":hits[1]/n,"recall_at_3":hits[3]/n,"recall_at_5":hits[5]/n,"mrr":sum(reciprocal)/n,"cases":rows}


def write_artifacts(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"rag_eval.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out_dir/"rag_eval.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["case_id","query","mode","first_relevant_rank","top_sources"])
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({**row,"top_sources":" | ".join(row["top_sources"])})
