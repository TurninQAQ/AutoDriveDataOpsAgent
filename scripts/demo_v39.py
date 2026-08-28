"""Small HTTP demo client. Start `autodrive-agent serve` first."""
from __future__ import annotations
import json, os
import httpx

BASE=os.getenv("AUTODRIVE_API_URL","http://127.0.0.1:8080")

def show(label,response):
    print(f"\n=== {label} ==="); print(json.dumps(response.json(),indent=2,ensure_ascii=False)); return response.json()

with httpx.Client(base_url=BASE,timeout=30) as c:
    show("health",c.get("/health"))
    show("READ diagnosis",c.post("/runs",json={"message":"task_A 为什么失败？"}))
    mixed=show("proposal",c.post("/runs",json={"message":"看看 task_A，如果只是优先级低就调到 5。"}))
    if mixed.get("status")=="WAITING_FOR_REVIEW":
        fp=mixed["pending_action"]["fingerprint"]
        show("approve + verified write",c.post(f"/runs/{mixed['run_id']}/approve",json={"fingerprint":fp}))
