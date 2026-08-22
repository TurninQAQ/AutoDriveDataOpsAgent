from deploy_ci_cloud_agentv2.agent.contracts import (
    CompletionContractCompiler,
    RequirementKind,
)
from deploy_ci_cloud_agentv2.agent.goals import DiagnoseTask, ExplainKnowledge, GoalDescriptor


def test_structured_goals_compile_without_raw_prompt_classification():
    descriptor = GoalDescriptor(
        1,
        (DiagnoseTask("g1", "task_A"), ExplainKnowledge("g2", "task_exclusive")),
    )
    contract = CompletionContractCompiler().compile(descriptor)
    assert [item.kind for item in contract.requirements_by_goal["g1"]] == [
        RequirementKind.TARGET_BINDING,
        RequirementKind.LIVE_TASK,
        RequirementKind.DIAGNOSTIC_CONTEXT,
    ]
    assert contract.requirements_by_goal["g2"][0].kind is RequirementKind.KNOWLEDGE
    assert contract.contract_fingerprint


def test_descriptor_revision_is_part_of_contract_identity():
    compiler = CompletionContractCompiler()
    one = compiler.compile(GoalDescriptor(1, (ExplainKnowledge("g1", "topic"),)))
    two = compiler.compile(GoalDescriptor(2, (ExplainKnowledge("g1", "topic"),)))
    assert one.contract_fingerprint != two.contract_fingerprint
