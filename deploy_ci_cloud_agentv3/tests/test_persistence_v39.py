from __future__ import annotations

import asyncio
import json

import pytest

from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore
from deploy_ci_cloud_agentv3.persistence.write_execution_store import SQLiteWriteExecutionStore
from deploy_ci_cloud_agentv3.persistence.run_store import RunStore


@pytest.mark.asyncio
async def test_concurrent_same_fingerprint_single_claim(tmp_path):
    store = SQLiteWriteExecutionStore(tmp_path / "state.sqlite")
    async def claim():
        return await store.claim("write_fp", fingerprint="fp", action="delete_task")
    results = await asyncio.gather(claim(), claim())
    assert sum(1 for claimed, _ in results if claimed) == 1


@pytest.mark.asyncio
async def test_persisted_dispatching_survives_store_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    first = SQLiteWriteExecutionStore(path)
    claimed, _ = await first.claim("write_fp", fingerprint="fp", action="resume_task")
    assert claimed
    second = SQLiteWriteExecutionStore(path)
    row = await second.get("write_fp")
    assert row is not None and row.status == "DISPATCHING" and row.mutation_attempted


@pytest.mark.asyncio
async def test_persisted_result_survives_store_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    first = SQLiteWriteExecutionStore(path)
    await first.claim("write_fp", fingerprint="fp", action="stop_task")
    await first.save_result("write_fp", status="VERIFIED", result={"status": "VERIFIED", "verified": True})
    second = SQLiteWriteExecutionStore(path)
    row = await second.get("write_fp")
    assert row is not None and row.result == {"status": "VERIFIED", "verified": True}


def test_audit_is_append_only_queryable_and_redacts(tmp_path):
    store = AuditStore(tmp_path / "state.sqlite")
    store.append("RUN_CREATED", {"api_key": "secret", "nested": {"Authorization": "Bearer x"}}, thread_id="t1", run_id="r1")
    rows = store.query(run_id="r1")
    assert len(rows) == 1
    assert rows[0]["payload"]["api_key"] == "[REDACTED]"
    assert rows[0]["payload"]["nested"]["Authorization"] == "[REDACTED]"


def test_run_store_persists_metadata(tmp_path):
    path = tmp_path / "state.sqlite"
    RunStore(path).create("r1", "t1")
    RunStore(path).update("r1", status="WAITING_FOR_REVIEW", pending_action={"fingerprint": "abc"})
    row = RunStore(path).get("r1")
    assert row["status"] == "WAITING_FOR_REVIEW"
    assert row["pending_action"]["fingerprint"] == "abc"
