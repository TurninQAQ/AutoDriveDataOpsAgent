from __future__ import annotations

"""Optional P1 API scaffold.

The V3.5 core remains AgentRuntime + LangGraph. This module intentionally keeps
service/API concerns out of the P0 graph and can be extended with durable run storage/SSE.
"""

from typing import Any


def create_app(runtime: Any):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install fastapi/uvicorn to enable the optional P1 API") from exc

    app = FastAPI(title="AutoDriveDataOpsAgent V3.5")

    class RunRequest(BaseModel):
        run_id: str
        message: str

    class ReviewRequest(BaseModel):
        decision: str
        fingerprint: str | None = None
        args: dict[str, Any] | None = None

    @app.post("/runs")
    async def create_run(body: RunRequest):
        return await runtime.start(body.run_id, body.message)

    @app.post("/runs/{run_id}/approve")
    async def approve(run_id: str, body: ReviewRequest):
        return await runtime.review(run_id, {"decision": "approve", "fingerprint": body.fingerprint or ""})

    @app.post("/runs/{run_id}/reject")
    async def reject(run_id: str, body: ReviewRequest):
        return await runtime.review(run_id, {"decision": "reject"})

    return app
