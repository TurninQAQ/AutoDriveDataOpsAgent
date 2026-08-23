"""Real OS-process tests for the single-instance Runtime boundary."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty

import pytest

try:
    import langgraph as _langgraph
except ModuleNotFoundError:  # pragma: no cover - compatibility test path
    _langgraph = None

_REAL_LANGGRAPH = _langgraph is not None and not getattr(
    _langgraph, "__v2_test_compat__", False
)

from deploy_ci_cloud_agentv2 import build_system_context, invoke, resume
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ResumeTask
from deploy_ci_cloud_agentv2.platform.facade import InMemoryPlatformFacade
from deploy_ci_cloud_agentv2.providers import ScriptedProvider
from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ResumeInput
from deploy_ci_cloud_agentv2.safety.runtime_lock import (
    RuntimeInstanceAlreadyActive,
    RuntimeInstanceLock,
)


def _descriptor() -> GoalDescriptor:
    return GoalDescriptor(1, (ResumeTask("g1", "task_A"),))


def _approval_values(pending) -> tuple[str, str, str]:
    return (
        pending.approval_request_id,
        pending.transaction_id,
        pending.fingerprint,
    )


class _CountingFacade(InMemoryPlatformFacade):
    def __init__(self, shared_count, **kwargs):
        super().__init__(**kwargs)
        self._shared_count = shared_count

    def resume_task(self, task_name: str):
        with self._shared_count.get_lock():
            self._shared_count.value += 1
        return super().resume_task(task_name)


class _BlockingCountingFacade(_CountingFacade):
    def __init__(self, started, release, **kwargs):
        super().__init__(**kwargs)
        self._started = started
        self._release = release

    def resume_task(self, task_name: str):
        with self._shared_count.get_lock():
            self._shared_count.value += 1
        self._started.set()
        if not self._release.wait(20):
            raise RuntimeError("cross-process test mutation was not released")
        return InMemoryPlatformFacade.resume_task(self, task_name)


class _CrashAfterDispatchFacade(_CountingFacade):
    def resume_task(self, task_name: str):
        with self._shared_count.get_lock():
            self._shared_count.value += 1
        # The graph has already durably recorded MutationStarted before this
        # method is entered.  _exit models a real hard process death and does
        # not run Runtime exception cleanup.
        os._exit(71)


def _resume_worker(
    db: str,
    runtime_root: str,
    approval: tuple[str, str, str],
    result_queue,
    shared_count,
    *,
    started=None,
    release=None,
    block: bool = False,
) -> None:
    facade_kwargs = {
        "tasks": {"task_A": {"state": "STOPPED", "revision": 1}},
        "shared_count": shared_count,
    }
    if block:
        facade = _BlockingCountingFacade(started, release, **facade_kwargs)
    else:
        facade = _CountingFacade(**facade_kwargs)
    context = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
        runtime_root=runtime_root,
    )
    try:
        result = asyncio.run(
            resume(
                thread_id="cross-process-live",
                resume_input=ResumeInput(
                    ApprovalDecision.APPROVE,
                    approval[0],
                    approval[1],
                    approval[2],
                ),
                system_context=context,
            )
        )
        result_queue.put(("result", result.status))
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


def _crash_worker(
    db: str,
    runtime_root: str,
    approval: tuple[str, str, str],
    shared_count,
) -> None:
    facade = _CrashAfterDispatchFacade(
        tasks={"task_A": {"state": "STOPPED", "revision": 1}},
        shared_count=shared_count,
    )
    context = build_system_context(
        ScriptedProvider([FinalCandidate("never returned", referenced_goal_ids=("g1",))]),
        read_facade=facade,
        durable_path=db,
        runtime_root=runtime_root,
    )
    asyncio.run(
        resume(
            thread_id="cross-process-live",
            resume_input=ResumeInput(
                ApprovalDecision.APPROVE,
                approval[0],
                approval[1],
                approval[2],
            ),
            system_context=context,
        )
    )


def _lock_holder(lock_path: str, ready, release) -> None:
    with RuntimeInstanceLock(lock_path):
        ready.set()
        release.wait(20)


def _next_result(queue, timeout: float = 10.0):
    try:
        return queue.get(timeout=timeout)
    except Empty as exc:  # pragma: no cover - turns a hung child into a useful failure
        raise AssertionError("child process did not report a result") from exc


def _create_paused_runtime(tmp_path: Path):
    db = tmp_path / "runtime.sqlite3"
    runtime_root = tmp_path / "runtime-root"
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    descriptor = _descriptor()
    context = build_system_context(
        ScriptedProvider([
            SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        ]),
        read_facade=facade,
        durable_path=db,
        runtime_root=runtime_root,
    )
    paused = asyncio.run(invoke("resume task_A", thread_id="cross-process-live", system_context=context))
    assert paused.status == "INTERRUPTED"
    return db, runtime_root, _approval_values(paused.pending_interrupt), context


@pytest.mark.real_langgraph
@pytest.mark.skipif(
    not _REAL_LANGGRAPH,
    reason="real LangGraph is unavailable; compatibility shim is active",
)
def test_live_mutation_owner_blocks_second_os_process_without_false_reconciliation(tmp_path):
    db, runtime_root, approval, verifier = _create_paused_runtime(tmp_path)
    process_context = mp.get_context("spawn")
    started = process_context.Event()
    release = process_context.Event()
    shared_count = process_context.Value("i", 0)
    queue = process_context.Queue()
    winner = process_context.Process(
        target=_resume_worker,
        args=(str(db), str(runtime_root), approval, queue, shared_count),
        kwargs={"started": started, "release": release, "block": True},
    )
    loser = None
    try:
        winner.start()
        assert started.wait(10), "winner did not reach the external mutation boundary"
        loser = process_context.Process(
            target=_resume_worker,
            args=(str(db), str(runtime_root), approval, queue, shared_count),
        )
        loser.start()
        loser.join(10)
        assert loser.exitcode == 0
        losing_result = _next_result(queue)
        assert losing_result[0:2] == ("error", "RuntimeInstanceAlreadyActive")
        assert "REQUIRES_RECONCILIATION" not in losing_result[2]
        assert shared_count.value == 1

        release.set()
        winner.join(20)
        assert winner.exitcode == 0
        winning_result = _next_result(queue)
        assert winning_result == ("result", "COMPLETED")

        events = verifier.event_store.for_thread("cross-process-live")
        event_types = [event.event_type for event in events]
        assert event_types.count("MutationStarted") == 1
        assert event_types.count("MutationResultRecorded") == 1
        assert "ReconciliationRequired" not in event_types
        assert "WriteReplayBlocked" not in event_types
    finally:
        release.set()
        for process in (loser, winner):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(5)


@pytest.mark.real_langgraph
@pytest.mark.skipif(
    not _REAL_LANGGRAPH,
    reason="real LangGraph is unavailable; compatibility shim is active",
)
def test_hard_process_death_after_mutation_started_reconciles_without_replay(tmp_path):
    db, runtime_root, approval, verifier = _create_paused_runtime(tmp_path)
    process_context = mp.get_context("spawn")
    shared_count = process_context.Value("i", 0)
    crashing = process_context.Process(
        target=_crash_worker,
        args=(str(db), str(runtime_root), approval, shared_count),
    )
    crashing.start()
    crashing.join(15)
    assert crashing.exitcode == 71
    assert shared_count.value == 1

    restarted = build_system_context(
        ScriptedProvider([FinalCandidate("must not replay", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=_CountingFacade(
            shared_count,
            tasks={"task_A": {"state": "STOPPED", "revision": 1}},
        ),
        durable_path=db,
        runtime_root=runtime_root,
    )
    result = asyncio.run(invoke("recover", thread_id="cross-process-live", system_context=restarted))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.terminal_outcome.code.value == "REQUIRES_RECONCILIATION"
    assert shared_count.value == 1
    event_types = [event.event_type for event in verifier.event_store.for_thread("cross-process-live")]
    assert "MutationStarted" in event_types
    assert "MutationResultRecorded" not in event_types
    assert "ReconciliationRequired" in event_types
    assert "WriteReplayBlocked" in event_types


def test_runtime_lock_releases_after_clean_exit_and_scopes_to_runtime_root(tmp_path):
    process_context = mp.get_context("spawn")
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    ready = process_context.Event()
    release = process_context.Event()
    holder = process_context.Process(
        target=_lock_holder,
        args=(str(root_a / "run" / "runtime.lock"), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeInstanceAlreadyActive):
            with RuntimeInstanceLock(root_a / "run" / "runtime.lock"):
                pass
        with RuntimeInstanceLock(root_b / "run" / "runtime.lock"):
            pass
        release.set()
        holder.join(10)
        assert holder.exitcode == 0
        with RuntimeInstanceLock(root_a / "run" / "runtime.lock"):
            pass
    finally:
        release.set()
        if holder.is_alive():
            holder.terminate()
            holder.join(5)


def test_single_instance_false_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="single_instance=false"):
        build_system_context(
            durable_path=tmp_path / "runtime.sqlite3",
            runtime_root=tmp_path / "runtime-root",
            single_instance=False,
        )
