from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from deploy_ci_cloud_agentv3 import __version__
from deploy_ci_cloud_agentv3.api.dependencies import AppServices
from deploy_ci_cloud_agentv3.api.events import EventBroker, encode_sse
from deploy_ci_cloud_agentv3.api.runs import router as runs_router
from deploy_ci_cloud_agentv3.api.reviews import router as reviews_router
from deploy_ci_cloud_agentv3.config import Settings
from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore
from deploy_ci_cloud_agentv3.persistence.run_store import RunStore


def create_app(services: AppServices | None = None) -> FastAPI:
    if services is not None:
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.services=services
            yield
    else:
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI):
            settings=Settings.from_env(); settings.ensure_dirs()
            from deploy_ci_cloud_agentv3.persistence.checkpoint import CheckpointerFactory
            from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
            from deploy_ci_cloud_agentv3.providers.qwen import QwenProvider
            async with CheckpointerFactory.open(settings.checkpoint_backend,path=settings.checkpoint_path) as saver:
                runtime=AgentRuntime.local(QwenProvider(),checkpointer=saver,audit_path=str(settings.db_path))
                app.state.services=AppServices(runtime,RunStore(settings.db_path),AuditStore(settings.db_path),EventBroker())
                yield

    app=FastAPI(title="AutoDriveDataOpsAgent API",version=__version__,lifespan=lifespan)
    app.include_router(runs_router); app.include_router(reviews_router)

    @app.get("/health")
    async def health(): return {"status":"ok","version":__version__}

    @app.get("/ready")
    async def ready(request: Request):
        if not hasattr(request.app.state,"services"): raise HTTPException(status_code=503,detail="runtime not ready")
        return {"status":"ready"}

    @app.get("/runs/{run_id}/events")
    async def events(run_id: str,request: Request):
        services=request.app.state.services
        try: row=services.run_store.get(run_id)
        except KeyError: raise HTTPException(status_code=404,detail="run not found")
        async def stream() -> AsyncIterator[str]:
            history=services.audit_store.query(run_id=run_id,limit=1000)
            for event in history: yield encode_sse(event)
            if row["status"] in {"COMPLETED","FAILED","UNCERTAIN"}: return
            async for event in services.broker.subscribe(run_id): yield encode_sse(event)
        return StreamingResponse(stream(),media_type="text/event-stream")
    return app


# Import-safe app object; runtime dependencies are resolved only in lifespan.
app=create_app()
