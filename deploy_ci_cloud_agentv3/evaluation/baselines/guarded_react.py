from .base import Baseline

class GuardedReAct(Baseline):
    name = "guarded_react"
    async def run(self, case, harness):
        return await harness.run_guarded(case)
