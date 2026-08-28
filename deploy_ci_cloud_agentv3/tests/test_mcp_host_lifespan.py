from __future__ import annotations

import contextlib

import pytest

pytest.importorskip("mcp")
from starlette.applications import Starlette

from deploy_ci_cloud_agentv3.mcp import server as server_module


class _FakeSessionManager:
    def __init__(self) -> None:
        self.running = False
        self.enter_count = 0
        self.exit_count = 0

    @contextlib.asynccontextmanager
    async def run(self):
        self.enter_count += 1
        self.running = True
        try:
            yield
        finally:
            self.running = False
            self.exit_count += 1


class _FakeMCPServer:
    def __init__(self) -> None:
        self.session_manager = _FakeSessionManager()
        self.app_built = False

    def streamable_http_app(self, **_kwargs):
        self.app_built = True
        return Starlette()


@pytest.mark.asyncio
async def test_parent_lifespan_runs_all_mounted_mcp_session_managers(monkeypatch):
    agent = _FakeMCPServer()
    runtime = _FakeMCPServer()

    monkeypatch.setattr(
        server_module,
        "build_mcp_servers",
        lambda _facade=None: (agent, runtime, object()),
    )

    app = server_module.create_app(object())
    assert agent.app_built is True
    assert runtime.app_built is True

    async with app.router.lifespan_context(app):
        assert agent.session_manager.running is True
        assert runtime.session_manager.running is True
        assert agent.session_manager.enter_count == 1
        assert runtime.session_manager.enter_count == 1

    assert agent.session_manager.running is False
    assert runtime.session_manager.running is False
    assert agent.session_manager.exit_count == 1
    assert runtime.session_manager.exit_count == 1
