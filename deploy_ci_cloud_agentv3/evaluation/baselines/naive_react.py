from .base import Baseline


class NaiveReAct(Baseline):
    name = "naive_react"

    async def run(self, case, harness):
        return await harness.run_naive(case)
