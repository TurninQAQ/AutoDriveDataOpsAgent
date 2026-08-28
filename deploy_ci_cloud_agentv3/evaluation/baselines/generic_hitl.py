from .base import Baseline


class GenericHITL(Baseline):
    name = "generic_hitl"

    async def run(self, case, harness):
        return await harness.run_generic_hitl(case)
