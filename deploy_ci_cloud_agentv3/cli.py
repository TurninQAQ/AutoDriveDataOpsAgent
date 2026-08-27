from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
from deploy_ci_cloud_agentv3.providers.qwen import QwenProvider


def _interrupt_payload(state):
    items = state.get("__interrupt__") if isinstance(state, dict) else None
    if not items:
        return None
    first = items[0]
    return getattr(first, "value", first)


async def _run(query: str) -> int:
    runtime = AgentRuntime.local(QwenProvider())
    thread_id = f"cli_{uuid.uuid4().hex}"
    state = await runtime.start(thread_id, query)
    while True:
        interrupt_payload = _interrupt_payload(state)
        if not interrupt_payload:
            print(json.dumps(state.get("final_response") or state, ensure_ascii=False, indent=2, default=str))
            return 0
        print(json.dumps(interrupt_payload, ensure_ascii=False, indent=2, default=str))
        raw = input("review [approve/reject/edit JSON]: ").strip()
        if raw.startswith("{"):
            decision = json.loads(raw)
        elif raw == "approve":
            decision = {"decision": "approve", "fingerprint": interrupt_payload.get("fingerprint")}
        else:
            decision = {"decision": raw or "reject"}
        state = await runtime.review(thread_id, decision)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDriveDataOpsAgent V3.5 CLI")
    parser.add_argument("query", nargs="*", help="natural-language request")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("request: ").strip()
    raise SystemExit(asyncio.run(_run(query)))


if __name__ == "__main__":
    main()
