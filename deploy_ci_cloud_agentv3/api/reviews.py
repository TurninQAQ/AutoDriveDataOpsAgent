from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .runs import apply_state
from .schemas import ApproveRequest, RejectRequest, EditReviewRequest

router=APIRouter()


def _pending(request: Request, run_id: str):
    try: row=request.app.state.services.run_store.get(run_id)
    except KeyError: raise HTTPException(status_code=404,detail="run not found")
    if row["status"] != "WAITING_FOR_REVIEW" or not row.get("pending_action"):
        raise HTTPException(status_code=409,detail="run is not waiting for review")
    return row,row["pending_action"]


def _check(expected: str | None, actual: str) -> None:
    if expected is not None and expected != actual:
        raise HTTPException(status_code=409,detail="review fingerprint mismatch")


@router.post("/runs/{run_id}/approve")
async def approve(run_id: str,payload: ApproveRequest,request: Request):
    row,pending=_pending(request,run_id); fp=str(pending["fingerprint"]); _check(payload.fingerprint,fp)
    services=request.app.state.services
    services.audit_store.append("REVIEW_APPROVED",{"fingerprint":fp},thread_id=row["thread_id"],run_id=run_id)
    state=await services.runtime.review(row["thread_id"],{"decision":"approve","fingerprint":fp})
    return await apply_state(request,run_id,state)


@router.post("/runs/{run_id}/reject")
async def reject(run_id: str,payload: RejectRequest,request: Request):
    row,pending=_pending(request,run_id); fp=str(pending["fingerprint"]); _check(payload.fingerprint,fp)
    services=request.app.state.services
    services.audit_store.append("REVIEW_REJECTED",{"fingerprint":fp,"reason":payload.reason},thread_id=row["thread_id"],run_id=run_id)
    state=await services.runtime.review(row["thread_id"],{"decision":"reject","reason":payload.reason})
    return await apply_state(request,run_id,state)


@router.post("/runs/{run_id}/edit")
async def edit(run_id: str,payload: EditReviewRequest,request: Request):
    row,pending=_pending(request,run_id); fp=str(pending["fingerprint"]); _check(payload.fingerprint,fp)
    services=request.app.state.services
    services.audit_store.append("REVIEW_EDITED",{"old_fingerprint":fp,"args":payload.args},thread_id=row["thread_id"],run_id=run_id)
    state=await services.runtime.review(row["thread_id"],{"decision":"edit","args":payload.args})
    return await apply_state(request,run_id,state)
