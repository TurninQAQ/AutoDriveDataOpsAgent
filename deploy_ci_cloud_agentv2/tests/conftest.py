"""Offline test compatibility for the pinned LangGraph dependency.

Production code never imports this module. It exists solely so the V2 test
suite can exercise the visible graph in review environments where the pinned
``langgraph`` wheel is unavailable. When LangGraph is installed, the real
package always wins and this shim is not installed into ``sys.modules``.
"""
from __future__ import annotations

import contextvars
import inspect
import sys
import types
from dataclasses import dataclass

try:  # pragma: no cover - exercised only in environments with LangGraph
    import langgraph.graph  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - the local review path
    START = "__start__"
    END = "__end__"

    @dataclass(frozen=True)
    class Command:
        resume: object | None = None

    @dataclass(frozen=True)
    class _InterruptValue:
        value: object

    class _GraphInterrupt(RuntimeError):
        def __init__(self, value):
            super().__init__("graph interrupted")
            self.value = value

    _resume_value = contextvars.ContextVar("langgraph_test_resume_value", default=None)
    _has_resume = contextvars.ContextVar("langgraph_test_has_resume", default=False)

    def interrupt(value):
        if _has_resume.get():
            return _resume_value.get()
        raise _GraphInterrupt(value)

    class InMemorySaver:
        """Tiny shared suspension store used only by the offline graph shim."""
        def __init__(self):
            self.paused = {}

    MemorySaver = InMemorySaver

    class _CompiledGraph:
        def __init__(self, nodes, edges, conditionals, entry, checkpointer=None):
            self._nodes = dict(nodes)
            self._edges = dict(edges)
            self._conditionals = dict(conditionals)
            self._entry = entry
            self._checkpointer = checkpointer or InMemorySaver()

        async def ainvoke(self, initial_state, config=None):
            config = config or {}
            configurable = config.get("configurable") or {}
            thread_id = configurable.get("thread_id")
            recursion_limit = int(config.get("recursion_limit", 100))

            resuming = isinstance(initial_state, Command)
            if resuming:
                if not thread_id or thread_id not in self._checkpointer.paused:
                    raise RuntimeError("no paused graph for thread")
                state, node = self._checkpointer.paused[thread_id]
                state = dict(state)
                resume_payload = initial_state.resume
            else:
                state = dict(initial_state)
                node = self._entry
                resume_payload = None

            steps = 0
            while node != END:
                steps += 1
                if steps > recursion_limit:
                    raise RuntimeError("Graph recursion limit exceeded")
                fn = self._nodes[node]
                token_has = _has_resume.set(bool(resuming))
                token_val = _resume_value.set(resume_payload)
                try:
                    try:
                        updates = fn(state)
                        if inspect.isawaitable(updates):
                            updates = await updates
                    except _GraphInterrupt as exc:
                        if not thread_id:
                            raise RuntimeError("interrupt requires configurable.thread_id") from exc
                        self._checkpointer.paused[thread_id] = (dict(state), node)
                        interrupted = dict(state)
                        interrupted["__interrupt__"] = (_InterruptValue(exc.value),)
                        return interrupted
                finally:
                    _resume_value.reset(token_val)
                    _has_resume.reset(token_has)
                resuming = False
                resume_payload = None
                if updates:
                    state.update(dict(updates))
                if node in self._conditionals:
                    chooser, mapping = self._conditionals[node]
                    route = chooser(state)
                    node = mapping[route]
                else:
                    node = self._edges[node]
            if thread_id:
                self._checkpointer.paused.pop(thread_id, None)
            return state

        def get_graph(self):
            edges = [types.SimpleNamespace(source=k, target=v) for k, v in self._edges.items()]
            for source, (_chooser, mapping) in self._conditionals.items():
                edges.extend(types.SimpleNamespace(source=source, target=target) for target in mapping.values())
            return types.SimpleNamespace(
                nodes={name: object() for name in self._nodes},
                edges=edges,
            )

    class StateGraph:
        def __init__(self, _state_type):
            self._nodes = {}
            self._edges = {}
            self._conditionals = {}
            self._entry = None

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, source, target):
            if source == START:
                self._entry = target
            else:
                self._edges[source] = target

        def add_conditional_edges(self, source, chooser, mapping):
            self._conditionals[source] = (chooser, dict(mapping))

        def compile(self, checkpointer=None, **_kwargs):
            if self._entry is None:
                raise RuntimeError("graph has no START edge")
            return _CompiledGraph(
                self._nodes, self._edges, self._conditionals, self._entry, checkpointer
            )

    pkg = types.ModuleType("langgraph")
    pkg.__v2_test_compat__ = True
    graph = types.ModuleType("langgraph.graph")
    graph.START = START
    graph.END = END
    graph.StateGraph = StateGraph
    types_mod = types.ModuleType("langgraph.types")
    types_mod.Command = Command
    types_mod.interrupt = interrupt
    checkpoint = types.ModuleType("langgraph.checkpoint")
    checkpoint_memory = types.ModuleType("langgraph.checkpoint.memory")
    checkpoint_memory.InMemorySaver = InMemorySaver
    checkpoint_memory.MemorySaver = MemorySaver

    pkg.graph = graph
    pkg.types = types_mod
    pkg.checkpoint = checkpoint
    checkpoint.memory = checkpoint_memory
    sys.modules.setdefault("langgraph", pkg)
    sys.modules.setdefault("langgraph.graph", graph)
    sys.modules.setdefault("langgraph.types", types_mod)
    sys.modules.setdefault("langgraph.checkpoint", checkpoint)
    sys.modules.setdefault("langgraph.checkpoint.memory", checkpoint_memory)
