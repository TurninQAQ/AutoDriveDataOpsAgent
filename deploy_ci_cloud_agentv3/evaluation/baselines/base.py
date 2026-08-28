from __future__ import annotations

from typing import Protocol

from deploy_ci_cloud_agentv3.evaluation.models import BenchmarkCase, BenchmarkOutcome


class Harness(Protocol):
    async def run_guarded(self, case: BenchmarkCase) -> BenchmarkOutcome: ...
    async def run_naive(self, case: BenchmarkCase) -> BenchmarkOutcome: ...
    async def run_generic_hitl(self, case: BenchmarkCase) -> BenchmarkOutcome: ...


class Baseline:
    name = "base"

    async def run(self, case: BenchmarkCase, harness: Harness) -> BenchmarkOutcome:
        raise NotImplementedError
