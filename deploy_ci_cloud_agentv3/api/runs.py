from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .schemas import CreateRunRequest

router = APIRouter()


def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    items = state.get("__interrupt__") if isinstance(state, dict) else None
    if not items:
        return None
    first = items[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else None


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


async def apply_state(request: Request, run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    services=request.app.state.services
    interrupt=_interrupt_payload(state)
    if interrupt:
        pending={k: interrupt.get(k) for k in ("action","args","before","artifact","reason","expected_effect","fingerprint") if k in interrupt}
        row=services.run_store.update(run_id,status="WAITING_FOR_REVIEW",pending_action=pending)
        services.audit_store.append("WAITING_FOR_REVIEW",pending,thread_id=row["thread_id"],run_id=run_id)
        await services.broker.publish(run_id,{"event_type":"WAITING_FOR_REVIEW","payload":pending})
        return _public_run(row)
    final=state.get("final_response")
    status="UNCERTAIN" if isinstance(final,dict) and final.get("status")=="write_uncertain" else "COMPLETED"
    row=services.run_store.update(run_id,status=status,final_response=final or {"status":"informational","message":"run completed"},pending_action=None)
    services.audit_store.append("FINAL_RESPONSE",row["final_response"] or {},thread_id=row["thread_id"],run_id=run_id)
    await services.broker.publish(run_id,{"event_type":"FINAL_RESPONSE","payload":row["final_response"]})
    return _public_run(row)


@router.post("/runs")
async def create_run(payload: CreateRunRequest, request: Request):
    services=request.app.state.services
    run_id=f"run_{uuid.uuid4().hex}"
    thread_id=payload.thread_id or f"thread_{uuid.uuid4().hex}"
    services.run_store.create(run_id,thread_id,"RUNNING")
    services.audit_store.append("RUN_CREATED",{"message":"run created"},thread_id=thread_id,run_id=run_id)
    await services.broker.publish(run_id,{"event_type":"RUN_CREATED","payload":{"thread_id":thread_id}})
    try:
        state=await services.runtime.start(thread_id,payload.message,run_id=run_id)
        return await apply_state(request,run_id,state)
    except Exception as exc:
        row=services.run_store.update(run_id,status="FAILED",error=f"{type(exc).__name__}: {exc}")
        services.audit_store.append("ERROR",{"error":row["error"]},thread_id=thread_id,run_id=run_id)
        await services.broker.publish(run_id,{"event_type":"ERROR","payload":{"error":row["error"]}})
        raise HTTPException(status_code=500,detail="agent run failed") from exc


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    try: return _public_run(request.app.state.services.run_store.get(run_id))
    except KeyError: raise HTTPException(status_code=404,detail="run not found")
