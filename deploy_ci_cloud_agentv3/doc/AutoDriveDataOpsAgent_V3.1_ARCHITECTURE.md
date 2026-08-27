# AutoDriveDataOpsAgent V3.1 总体重构方案

> **版本定义：Guarded ReAct + Standard MCP**
>
> 本方案用于将当前 AutoDriveDataOpsAgent V2 从“可靠性机制过重、Tool 接入层标准化不足”的状态，重构为适合 2026 秋招 Agent / LLM Application / AI Agent Engineer 岗位展示的 V3.1。
>
> 核心目标不是构建银行级分布式 Agent Runtime，而是：
>
> 1. Agent 核心范式正确；
> 2. 架构可以在 3～5 分钟内讲清楚；
> 3. Tool 接入、Function Calling、HITL 等关键概念使用标准实现；
> 4. 保留 DataOps WRITE 场景中真正有价值的可靠性能力；
> 5. 保留模拟 AutoDrive 平台作为完整业务载体；
> 6. 为后续 Benchmark、FastAPI、Demo 留出清晰扩展点。

---

# 1. 最终架构定义

AutoDriveDataOpsAgent V3.1 是一个基于 LangGraph 的 **Single-Agent Guarded ReAct** 系统。

平台的 READ、RAG、Prepare、Proposal 与 WRITE 能力统一通过标准 MCP 接入，并通过 Agent / Runtime 两类 Capability Profile 隔离模型可见能力与真实副作用能力。

模型通过 Native Function Calling 自主调用 READ、RAG、Prepare 和无副作用 Proposal Tool。

涉及平台修改时：

```text
Proposal
→ PendingAction
→ HITL Review
→ Approve
→ Precondition Recheck
→ WriteService
→ Runtime MCP WRITE
→ Observe Again
→ Post-write Verification
→ FinalGuard
```

核心原则：

> **LLM 管语义，Runtime 管确定性。**

> **MCP 管 Tool 接入，HITL 管执行授权。**

> **Proposal ≠ Execution。**

> **API success ≠ business success。**

---

# 2. 四层核心架构

整个项目以后统一按照四层理解。

```text
Reasoning
    ↓
Single-Agent ReAct

Integration
    ↓
Standard MCP + Native Function Calling

Authority
    ↓
Proposal + HITL + WriteService

Reliability
    ↓
Precondition + Idempotency + Verification
```

其中：

- LangGraph：Workflow orchestration
- LLM：语义理解与下一步动作选择
- MCP：平台能力发现、Schema 和调用协议
- Function Calling：LLM 生成 Tool name + arguments
- HITL：真实 WRITE 前的人类授权
- WriteService：审批后的确定性执行
- Platform Backend：真实业务逻辑
- Verification：判断业务状态是否真的达到预期

---

# 3. 总体架构图

```text
                              User
                               │
                               ▼
                     ┌──────────────────┐
                     │ LangGraph Agent  │
                     │ Single ReAct     │
                     └────────┬─────────┘
                              │
                    Native Function Calling
                              │
                      Agent MCP Client
                              │
                         /mcp/agent
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
        READ               PREPARE             PROPOSAL
          │                   │                   │
 get_task_detail       prepare_task_spec   propose_priority
 get_gpu_pool                             propose_resume
 get_queue_state                          propose_stop
 diagnose_task                            propose_delete
 search_knowledge                         propose_submit
          │                   │                   │
          └────────────── ToolResult ─────────────┘
                              │
                    ProposalResult ?
                       /              \
                     NO                YES
                     │                  │
                     ▼                  ▼
                   Agent          PendingAction
                                      │
                                 Before / After
                                 Precondition
                                 Fingerprint
                                      │
                                      ▼
                                  HITL Review
                                  /        \
                              Reject      Approve
                                │            │
                                ▼            ▼
                              Agent      WriteService
                                             │
                                  validate approval
                                             │
                                  recheck precondition
                                             │
                                   idempotency check
                                             │
                                  Runtime MCP Client
                                             │
                                       /mcp/runtime
                                             │
                                      REAL WRITE
                                             │
                                             ▼
                                     Platform Backend
                                             │
                                      Observe Again
                                             │
                                         Verify
                                             │
                                          Audit
                                             │
                                             ▼
                                           Agent
                                             │
                                       FinalGuard
                                             │
                                             ▼
                                            END
```

---

# 4. LangGraph：只保留四个核心 Node

最终 Graph：

```text
agent
model_tools
review
execute_write
```

不再保留：

```text
intent_classifier
planner
replanner
rag_node
write_guard
revalidate_write
action_verify
operational_goal_verify
completion_gate
evidence_node
execution_claim_node
```

## 4.1 Node 划分依据

Node 不按照业务功能拆，而按照“谁掌握当前控制权”拆。

```text
agent
→ LLM Decision Authority

model_tools
→ Safe Tool Execution Authority

review
→ Human Approval Authority

execute_write
→ Privileged Runtime Authority
```

因此：

- `agent`：模型负责 Reason
- `model_tools`：执行模型有权限调用的 Tool
- `review`：LangGraph interrupt，把控制权交给 Human
- `execute_write`：审批后由 deterministic Runtime 执行真实副作用

Precondition、Idempotency、Mutation、Verification、Audit 没有新的控制权切换，因此不单独做 Graph Node，而是放进 WriteService。

这就是：

> **Thin Graph, Rich Service。**

---

# 5. Graph 节点切换条件

所有 Edge 基于结构化状态，不做 READ / WRITE Intent Classifier。

## 5.1 START → agent

所有请求统一：

```text
User
 ↓
Agent
```

不提前分类：

```text
READ?
WRITE?
MIXED?
```

Agent 自己根据 ReAct Loop 决定下一步。

---

## 5.2 agent → model_tools

条件：

```python
assistant_message.tool_calls != []
```

即 LLM 通过 Native Function Calling 产生 ToolCall。

---

## 5.3 agent → FinalGuard → END

条件：

```text
没有 tool_calls
+
生成 FinalResponse
```

最终回答必须经过 FinalGuard。

---

## 5.4 model_tools → agent

ToolResult 为普通结果，例如：

```text
Observation
PreparedArtifact
ToolError
```

例如：

```text
get_task_detail
↓
Observation
↓
Agent
```

---

## 5.5 model_tools → review

Tool 返回：

```text
ProposalResult
```

例如：

```text
propose_set_task_priority
↓
ProposalResult
↓
review
```

不依赖 LLM 再判断“这是 WRITE”。

---

## 5.6 review → execute_write

条件：

```text
decision = approve
+
approved fingerprint == pending_action fingerprint
```

---

## 5.7 review → agent

条件：

```text
decision = reject
```

平台不产生 mutation。

---

## 5.8 review → review

条件：

```text
decision = edit
```

Edit 后：

```text
old fingerprint invalid
↓
rebuild PendingAction
↓
new fingerprint
↓
重新 Review
```

禁止“批准 A、执行 B”。

---

## 5.9 execute_write → agent

无论：

```text
VERIFIED
FAILED
PRECONDITION_FAILED
VERIFICATION_FAILED
UNKNOWN_OUTCOME
```

全部回 Agent。

Agent基于结构化 WriteResult 决定：

```text
继续 READ
重新 Proposal
或 Final
```

---

# 6. 删除 READ / WRITE Intent Classifier

V3.1 不做：

```text
User
 ↓
Intent Classifier
 ↓
READ / WRITE
```

原因：

1. 多一次 LLM 调用；
2. 它无法成为安全边界；
3. 混合请求很难提前准确分类；
4. ReAct 本身就负责动态决定下一步。

例如用户：

```text
看看 task_A 怎么了，如果只是优先级低就帮我调高。
```

正确流程：

```text
agent
 ↓
get_task_detail
 ↓
model_tools
 ↓
agent
 ↓
get_queue_state
 ↓
model_tools
 ↓
agent
 ↓
search_knowledge
 ↓
model_tools
 ↓
agent
 ↓
propose_set_task_priority
 ↓
model_tools
 ↓
review
```

Runtime 不需要理解用户自然语言属于 READ 还是 WRITE。

---

# 7. V2 MCP 问题与 V3.1 MCP 标准化

## 7.1 V2

V2 当前 Tool 调用本质大致为：

```text
Agent Runtime
 ↓
MCPPlatformFacade
 ↓
手写 HTTP JSON-RPC
 ↓
POST /mcp
 ↓
custom tools/call dispatcher
 ↓
Platform Service
```

这是一个自定义 MCP-like Tool Gateway。

它不应该在简历中声称为“标准 MCP Client / Server”。

---

## 7.2 V3.1

改成真正标准 MCP：

```text
Official MCP Client
 ↓
Standard MCP Transport
 ↓
Official MCP Server
 ↓
tools/list
tools/call
 ↓
Platform Service
```

Transport：

```text
Streamable HTTP
```

原因：

```text
Agent Runtime
↔
Platform Tool Service
```

本身就是服务边界。

MCP 层只负责：

```text
Tool Discovery
Tool Schema
Tool Invocation
Tool Result Protocol
Transport
```

不负责：

```text
HITL
Precondition
Write Authorization
Business Verification
```

---

# 8. MCP：一套实现、两个 Capability Profile

不是两个独立工程。

统一：

```text
AutoDrive MCP Service
│
├── Shared Tool Code
├── Shared Pydantic Schema
├── Shared Platform Service
│
├── Agent Profile
│   └── /mcp/agent
│
└── Runtime Profile
    └── /mcp/runtime
```

---

# 9. `/mcp/agent`

只提供模型允许发现的能力。

## READ

```text
get_task_detail
get_gpu_pool
get_queue_state
diagnose_task
search_knowledge
```

## PREPARE

```text
prepare_task_spec
```

## PROPOSAL

```text
propose_set_task_priority
propose_resume_task
propose_stop_task
propose_delete_task
propose_submit_task
```

不包含：

```text
set_task_priority
resume_task
stop_task
delete_task
submit_task
```

因此即使 Agent 端直接：

```python
bind_tools(all_tools)
```

也拿不到真实 WRITE Tool。

安全边界建立在 Tool Discovery 层。

---

# 10. `/mcp/runtime`

只供 Runtime / WriteService 使用。

包含：

## READ

```text
get_task_detail
get_queue_state
...
```

## WRITE

```text
set_task_priority
resume_task
stop_task
delete_task
submit_task
```

真实 WRITE Tool 永远不 bind 给 LLM。

---

# 11. Tool 最终分成四类

## 11.1 READ Tool

真实执行，无平台副作用。

```text
get_task_detail
get_gpu_pool
get_queue_state
diagnose_task
search_knowledge
```

---

## 11.2 PREPARE Tool

真实执行计算，但不修改平台。

```text
prepare_task_spec
```

负责：

```text
TaskDraft
↓
Platform Defaults
↓
TaskSpec
↓
Schema Validation
↓
YAML Artifact
```

---

## 11.3 PROPOSAL Tool

模型可直接 Function Call，但没有平台 Mutation 能力。

```text
propose_set_task_priority
propose_resume_task
propose_stop_task
propose_delete_task
propose_submit_task
```

例如：

```python
propose_set_task_priority(
    task_name="task_A",
    priority=5
)
```

返回：

```json
{
  "kind": "ACTION_PROPOSAL",
  "action": "set_task_priority",
  "args": {
    "task_name": "task_A",
    "priority": 5
  },
  "reason": "...",
  "expected_effect": "..."
}
```

平台状态不变化。

---

## 11.4 WRITE Tool

真实平台副作用。

```text
set_task_priority
resume_task
stop_task
delete_task
submit_task
```

只有 WriteService 能调用。

---

# 12. Proposal Tool 与 WRITE Tool 共享参数 Schema

例如：

```python
class SetTaskPriorityArgs(BaseModel):
    task_name: str
    priority: int
```

共享关系：

```text
                 SetTaskPriorityArgs
                    /           \
                   /             \
                  ▼               ▼
propose_set_task_priority   set_task_priority
        SAFE                     WRITE
```

因此：

> 两个 Tool，不同权限；一份参数契约。

同理：

```text
DeleteTaskArgs
├── propose_delete_task
└── delete_task
```

---

# 13. Native Function Calling

这是 V3.1 的 P0 改造。

## 13.1 V2 问题

V2 Provider 更接近：

```text
LLM
 ↓
response_format=json_schema
 ↓
AgentDecision JSON
 ↓
SINGLE_TOOL_CALL / READ_TOOL_BATCH / FINAL...
```

它能够实现 Tool Selection，但不是标准的 Native Function Calling。

---

## 13.2 V3.1

改成：

```text
MCP Server
 ↓
tools/list
 ↓
Tool name / description / inputSchema
 ↓
MCP → Provider Tool Adapter
 ↓
Qwen native tools=[...]
 ↓
LLM
 ↓
native tool_calls
 ↓
model_tools
 ↓
MCP tools/call
```

最终真正成为：

```text
Reason
→ Native ToolCall
→ MCP Tool
→ Observation
→ Reason
```

---

# 14. MCP 与 Function Calling 的关系

以后统一这样解释：

```text
MCP
=
有哪些 Tool？
Schema 是什么？
Tool 怎么被 Client 调用？

Function Calling
=
LLM 根据 Tool Schema 输出：
tool name + arguments

Runtime
=
真正执行 Tool
```

完整：

```text
MCP tools/list
      ↓
Function Calling Tool Schema
      ↓
LLM native tool_calls
      ↓
Runtime
      ↓
MCP tools/call
```

MCP 不替代 Function Calling。

---

# 15. Provider 层重构

建议：

```text
providers/
├── base.py
├── qwen.py
├── scripted.py
└── tool_adapter.py
```

统一接口：

```python
class ModelProvider:

    async def invoke(
        self,
        messages,
        tools,
    ) -> AssistantMessage:
        ...
```

这里 `tools` 来自：

```text
Agent MCP Client
 ↓
tools/list
```

然后由 `tool_adapter.py` 转换成模型 Provider 所要求的 Native Function Calling Schema。

---

# 16. 删除 V2 Agent Decision DSL

V3.1 不再使用：

```text
READ_ACTION
READ_TOOL_BATCH
SINGLE_TOOL_CALL
FINAL_CANDIDATE
```

READ：

```text
native tool_call
```

Proposal：

```text
native tool_call:
propose_delete_task(...)
```

Final：

```text
assistant final response
```

这样 Agent Protocol 更接近标准 ReAct。

---

# 17. READ 并行

删除：

```text
READ_TOOL_BATCH
```

模型如果一次返回：

```text
get_task_detail
get_gpu_pool
get_queue_state
```

Executor：

```python
await asyncio.gather(...)
```

即可。

Policy：

```text
READ + READ + READ
→ 可以并行
```

Proposal：

```text
Proposal
→ 必须独占当前 ToolCall Round
```

不要：

```text
get_task_detail
+
propose_delete_task
```

同轮执行。

也不要：

```text
propose_stop_task
+
propose_delete_task
```

同轮执行。

建议规则：

```python
if proposal_count == 0:
    execute_normal_calls()

elif proposal_count == 1 and len(tool_calls) == 1:
    execute_proposal()

else:
    reject_batch_and_return_agent()
```

这样一个 Agent Run 同一时刻只存在一个 PendingAction。

---

# 18. Proposal → PendingAction

LLM / Proposal Tool 只负责：

```text
action
args
reason
expected_effect
```

Runtime 补安全字段：

```python
PendingAction(
    proposal_id=...,
    action=...,
    args=...,

    before=...,
    artifact=...,

    precondition=...,
    fingerprint=...
)
```

禁止 LLM 自己生成：

```text
fingerprint
approval status
idempotency key
precondition token
```

这些必须属于 Runtime Authority。

---

# 19. HITL

进入 Review 前必须已经生成完整 PendingAction。

示例：

```text
Task: task_A

Current
────────────────
Priority: 3

Proposal
────────────────
Priority: 3 → 5

Reason
────────────────
当前任务处于 QUEUED，
根据 Priority Policy，
HIGH = 5。

Expected Effect
────────────────
提高后续 Scheduler 调度优先级
```

然后：

```text
fingerprint
=
hash(
  action
  + canonical_args
  + artifact
  + precondition
)
```

LangGraph：

```text
review
 ↓
interrupt(...)
```

用户：

```text
Approve
Reject
Edit
```

---

# 20. Approve 后 LLM 退出执行链

错误：

```text
Approve
 ↓
Agent
 ↓
LLM重新生成 ToolCall
 ↓
execute
```

正确：

```text
Proposal
 ↓
freeze
 ↓
fingerprint
 ↓
HITL
 ↓
Approve exact fingerprint
 ↓
execute_write
 ↓
WriteService.execute(frozen_action)
```

用户批准什么，Runtime 就执行什么。

---

# 21. WriteService

WriteService 是整个 V3.1 的确定性 WRITE 核心。

建议：

```python
class WriteService:

    async def execute(self, action):

        self.validate_approval(action)

        await self.recheck_precondition(action)

        self.check_idempotency(action)

        raw = await self.runtime_mcp.call_tool(
            action.name,
            action.args,
        )

        verified = await self.verify(
            action,
            raw,
        )

        await self.audit(
            action,
            verified,
        )

        return verified
```

内部流程：

```text
Validation
↓
Precondition
↓
Idempotency
↓
WRITE
↓
Observe Again
↓
Verify
↓
Audit
```

WriteService 与 MCP 的关系：

```text
WriteService
=
什么时候允许 WRITE

MCP
=
WRITE Tool 怎么被调用
```

---

# 22. Precondition Recheck

用途：

> 防止 stale approval / TOCTOU。

例如：

```text
Proposal 时：

task_A
status=QUEUED
priority=3
```

用户看了 20 秒。

平台变成：

```text
status=RUNNING
priority=4
```

Approve 后：

```text
WriteService
 ↓
MCP READ
 ↓
current platform state
 ↓
compare frozen precondition
```

不一致：

```text
PRECONDITION_FAILED
```

不得调用真实 WRITE。

然后：

```text
Agent
 ↓
重新 READ
 ↓
重新 Reason
 ↓
如果仍需要 WRITE
 ↓
新的 Proposal
 ↓
新的 HITL
```

Checkpoint 不能替代 Precondition。

---

# 23. Checkpoint 与 Precondition

必须永远区分：

## Checkpoint

```text
Agent Workflow State
```

例如：

```text
messages
tool_results
pending_action
graph position
interrupt/resume
```

作用：

> 恢复 Agent workflow。

---

## Precondition

```text
Real Platform Business State
```

例如：

```text
task_A
status=QUEUED
priority=3
```

作用：

> 判断用户审批期间真实平台是否发生变化。

恢复流程：

```text
恢复 Checkpoint
+
重新 READ Platform
+
比较 Precondition
```

---

# 24. Post-write Verification

核心原则：

> **API success ≠ business success。**

例如：

```text
MCP set_task_priority
 ↓
CallToolResult OK
```

只能说明：

```text
调用成功返回
```

不能说明：

```text
priority 已经真的等于 5
```

必须：

```text
MCP get_task_detail
 ↓
priority == 5 ?
```

只有成立：

```python
WriteResult(
    status="VERIFIED",
    verified=True,
)
```

Agent 才允许告诉用户：

```text
已成功修改。
```

---

# 25. WRITE Result

建议状态：

```python
WriteStatus = Literal[
    "VERIFIED",
    "FAILED",
    "PRECONDITION_FAILED",
    "VERIFICATION_FAILED",
    "UNKNOWN_OUTCOME",
    "REJECTED",
]
```

结构：

```python
WriteResult(
    id=...,
    action=...,
    status=...,
    verified=...,
    before=...,
    after=...,
)
```

---

# 26. 幂等与 UNKNOWN_OUTCOME

V3.1 不讲：

```text
Exactly-once
ExecutionClaim
Lease Ownership
```

只保留四个概念：

```text
idempotency_key

single mutation attempt

UNKNOWN_OUTCOME

reconcile_by_read
```

---

## 26.1 正常情况

```text
Approve
 ↓
idempotency_key=A
 ↓
WRITE 一次
 ↓
Verify
```

---

## 26.2 Unknown Outcome

例如：

```text
submit_task
 ↓
请求已发出
 ↓
connection reset
```

无法判断：

```text
服务端到底有没有执行
```

此时：

```text
UNKNOWN_OUTCOME
```

绝对不直接 retry WRITE。

而是：

```text
MCP READ
 ↓
Task / DAG / Queue
 ↓
判断第一次 WRITE 是否已经生效
```

如果确认：

```text
VERIFIED
```

无法确认：

```text
UNKNOWN_OUTCOME
```

面试表达：

> 对可能已经产生副作用的 Mutation 不进行盲目 retry，而是进入 UNKNOWN_OUTCOME，通过真实平台 READ-back reconciliation 判断实际结果。

---

# 27. FinalGuard

删除 V2：

```text
GoalDescriptor
CompletionContract
Evidence Qualification
GoalOutcome
ResponseCompletionGate
```

改成简单结构化 Final：

```python
class FinalResponse(BaseModel):
    status: Literal[
        "informational",
        "write_verified",
        "write_failed",
        "write_not_executed",
        "write_uncertain",
    ]

    write_result_id: str | None
    message: str
```

FinalGuard：

```python
if response.status == "write_verified":
    assert write_result is not None
    assert write_result.verified is True
```

不再对自然语言做：

```text
“这句话是不是在声称成功？”
```

这种 NLP 判断。

---

# 28. RAG

V2 已经存在真实：

```text
Chunking
BM25
Vector Score
Hybrid Fusion
Top-K
```

但默认 Vector 侧是：

```text
feature hashing
```

V3.1 建议升级成真实 Dense Embedding：

```text
Knowledge Base
 ↓
Chunking
 ↓
 ┌──────────────┐
 │              │
BM25       Dense Embedding
 │              │
 └──────┬───────┘
        ↓
 Fusion / RRF
        ↓
      Top-K
```

不需要为了简历引入：

```text
Milvus
Qdrant
ElasticSearch
```

当前知识库规模用内存向量 / 轻量本地索引即可。

RAG 在 Agent 中仍然只是：

```text
search_knowledge
```

一个普通 MCP READ Tool。

不要单独创建：

```text
RAG Node
Retrieval Agent
Knowledge Agent
```

---

# 29. RAG 调用链

```text
Agent
 ↓
Native Function Call
 ↓
search_knowledge
 ↓
Agent MCP Client
 ↓
/mcp/agent
 ↓
RAG Service
 ↓
BM25 + Dense
 ↓
Fusion
 ↓
Top-K
 ↓
ToolResult
 ↓
AgentState
 ↓
ContextBuilder
 ↓
Agent
```

---

# 30. Task Pipeline

不加入 General Planner。

用户：

```text
对 /data/abc 创建训练任务。
```

流程：

```text
User
 ↓
Agent
 ↓
READ
 ↓
prepare_task_spec
 ↓
TaskDraft
 ↓
Platform Defaults
 ↓
TaskSpec
 ↓
Schema Validation
 ↓
YAML Artifact
 ↓
propose_submit_task
 ↓
HITL Review YAML
 ↓
WriteService
 ↓
Runtime MCP
 ↓
submit_task
 ↓
Post-write Verification
```

这里：

```text
TaskPreparationService
```

是 deterministic domain service。

不是：

```text
Planner Agent
```

LLM只提取用户明确表达的信息。

平台默认值、合法性校验必须由 deterministic service 决定。

---

# 31. prepare_task_spec

职责：

```text
Natural Language
 ↓
TaskDraft
 ↓
Inject Platform Defaults
 ↓
TaskSpec
 ↓
Validate
 ↓
YAML Artifact
```

禁止模型自己猜：

```text
GPU ID
Image Tag
Timeout
Max Active Runs
Memory Default
Scheduler Default
```

这些必须来自 Platform Schema / deterministic defaults。

---

# 32. propose_submit_task

只能绑定已经准备好的 Artifact。

正确：

```text
prepare_task_spec
 ↓
PreparedTaskArtifact(id/hash)
 ↓
propose_submit_task(
    artifact_id=...
)
```

禁止：

```text
Agent 临时生成 config
 ↓
直接 propose submit
```

HITL Review 必须展示最终 YAML Artifact。

---

# 33. Persistence 简化

V2 Persistence 过重。

V3.1 只保留两个核心概念。

## 33.1 LangGraph Checkpoint

负责：

```text
messages
tool_results
pending_action
graph position
interrupt/resume
```

即：

> Workflow Persistence。

---

## 33.2 AuditStore

负责：

```text
Proposal
Approval
Write Action
Write Result
Verification
Error
```

即：

> Business Audit。

不再维护多套 durability authority：

```text
LangGraph checkpoint
+
自研 Runtime checkpoint
+
ExecutionClaimStore
+
Lease Ownership
+
复杂 Event Tail Protocol
```

---

# 34. ContextBuilder

保留 ContextBuilder，但大幅瘦身。

输入：

```text
System Prompt
+
必要 recent messages
+
相关 recent ToolResults
+
pending_action
+
last_write_result
+
optional summary
```

不要：

```text
所有 MCP raw JSON
所有历史 Observation
所有 Audit Records
所有 Runtime State
```

直接塞 Prompt。

Tool Result 必须先变成统一结构，再进入 Agent Context。

---

# 35. AgentState

建议：

```python
class AgentState(TypedDict):
    thread_id: str

    messages: list
    tool_results: list[ToolResult]

    pending_action: PendingAction | None
    last_write_result: WriteResult | None

    prepared_artifact: PreparedArtifact | None

    step_count: int
```

删除：

```text
goal_descriptor
completion_contract
goal_outcomes
evidence
gate_feedback
execution_claim
...
```

---

# 36. 模拟 Platform 保持

V3.1 继续使用：

```text
Mock / Simulated AutoDrive Training Platform
```

包括：

```text
Task
GPU Pool
Queue
Airflow-like DAG
Docker-like Runtime
Knowledge Base
```

真实 Adapter 可以继续保留：

```text
Airflow Gateway
Docker Gateway
GPU Runtime Adapter
```

但 README 必须明确：

> 当前系统验证范围为 single-node mock / simulated AutoDrive platform。

不把项目宣传成真实生产集群。

---

# 37. Observability

保留轻量：

```text
Structured Agent Events
Tool Call Trace
Proposal
Approval
WriteResult
VerificationResult
JSONL / SQLite Audit
```

暂不引入：

```text
OpenTelemetry
Jaeger
Prometheus
LangSmith
```

除非最终 Demo 或部署确实需要。

---

# 38. Benchmark

当前阶段：

> **暂时不重构。**

保留 V2 现有：

```text
evaluation/
metrics
audit evaluator
runner
```

等 V3.1 主架构完成以后，再单独设计真正：

```text
Naive ReAct
vs
Generic HITL
vs
Guarded ReAct
```

Benchmark。

当前不要让 Benchmark 重构影响核心 Agent 架构进度。

---

# 39. FastAPI

FastAPI 放 P1。

核心 Agent 完成后再：

```text
FastAPI
 ↓
AgentRuntime
 ↓
LangGraph
```

接口建议：

```text
POST /runs

GET /runs/{run_id}

POST /runs/{run_id}/approve

POST /runs/{run_id}/reject

GET /runs/{run_id}/events
```

SSE：

```text
agent
tool_call
observation
proposal
waiting_review
write_result
final
```

---

# 40. 推荐目录

```text
deploy_ci_cloud_agentv3/
│
├── agent/
│   ├── graph.py
│   ├── runtime.py
│   ├── state.py
│   ├── context_builder.py
│   ├── final_guard.py
│   └── prompts.py
│
├── mcp/
│   ├── server.py
│   ├── client.py
│   ├── registry.py
│   ├── profiles.py
│   ├── schemas.py
│   │
│   └── tools/
│       ├── read.py
│       ├── prepare.py
│       ├── proposal.py
│       └── write.py
│
├── providers/
│   ├── base.py
│   ├── qwen.py
│   ├── scripted.py
│   └── tool_adapter.py
│
├── services/
│   ├── write_service.py
│   ├── task_preparation.py
│   ├── verification.py
│   └── audit.py
│
├── models/
│   ├── tool_result.py
│   ├── proposal.py
│   ├── pending_action.py
│   ├── write_result.py
│   ├── artifact.py
│   └── final_response.py
│
├── rag/
│   ├── chunker.py
│   ├── bm25.py
│   ├── dense.py
│   └── hybrid.py
│
├── platform_backend/
│   └── reuse V2
│
├── evaluation/
│   └── temporarily keep V2 design
│
├── api/
│   └── app.py
│
└── tests/
```

---

# 41. 开发顺序

## Phase 0 — V3 Skeleton / Cleanup

创建：

```text
deploy_ci_cloud_agentv3/
```

V2 保留不动。

优先复用：

```text
platform_backend
domain services
test fixtures
```

V3 主线不再迁移：

```text
GoalDescriptor
CompletionContract
Evidence system
ExecutionClaim generalized protocol
Complex Completion Gate
```

---

## Phase 1 — Standard MCP

先迁移 READ：

```text
get_task_detail
get_gpu_pool
get_queue_state
diagnose_task
search_knowledge
```

实现：

```text
/mcp/agent
/mcp/runtime
```

验证：

```text
tools/list
tools/call
```

先不要碰 WRITE。

---

## Phase 2 — Native Function Calling + Minimal ReAct

实现：

```text
MCP tools/list
 ↓
Provider Tool Adapter
 ↓
Native Function Calling
 ↓
model_tools
 ↓
MCP tools/call
 ↓
Observation
 ↓
Agent
```

跑通 Demo：

```text
task_A 为什么失败？
```

此时 Graph 可以只跑：

```text
agent
↔
model_tools
```

---

## Phase 3 — Proposal + HITL + set_task_priority

实现：

```text
propose_set_task_priority
```

以及 Runtime：

```text
set_task_priority
```

跑通：

```text
READ
 ↓
Proposal
 ↓
PendingAction
 ↓
HITL
 ↓
Precondition
 ↓
Runtime MCP WRITE
 ↓
Verification
 ↓
Final
```

这是整个 V3.1 最核心阶段。

---

## Phase 4 — Complete WRITE

加入：

```text
resume
stop
delete
```

全部复用：

```text
Proposal
→ Review
→ WriteService
```

不增加 Graph Node。

---

## Phase 5 — Pipeline Submit

加入：

```text
prepare_task_spec
TaskDraft
TaskSpec
YAML
propose_submit_task
submit_task
```

---

## Phase 6 — Reliability Closure

补齐：

```text
fingerprint
edit invalidation
Precondition
idempotency_key
single mutation attempt
UNKNOWN_OUTCOME
reconcile_by_read
Post-write Verification
FinalGuard
```

---

## Phase 7 — Persistence Cleanup

最终收敛成：

```text
LangGraph Checkpoint
+
AuditStore
```

删除 V2 过重 persistence machinery。

---

## Phase 8 — RAG Dense Upgrade

将默认：

```text
feature hashing vector
```

升级成：

```text
Dense Embedding
```

保留：

```text
BM25 + Dense + Fusion
```

不引入 Vector DB。

---

## Phase 9 — FastAPI / Demo

最后再服务化。

Benchmark 在主架构稳定以后另开阶段设计。

---

# 42. P0 / P1 / 暂不做

## P0

```text
4-node Guarded ReAct Graph

Standard MCP

Agent / Runtime MCP Capability Profile

Native Function Calling

READ / PREPARE / PROPOSAL / WRITE Boundary

Proposal HITL

Frozen PendingAction

Fingerprint

Precondition Recheck

WriteService

Post-write Verification

FinalGuard

TaskSpec / YAML Pipeline
```

---

## P1

```text
Persistence Simplification

UNKNOWN_OUTCOME Reconciliation

RAG Real Dense Embedding

FastAPI

Demo UI
```

---

## 暂不做

```text
General Planner

Multi-Agent

Intent Classifier

Real Production Cluster

Vector DB

OpenTelemetry

LangSmith

Complex Exactly-once Protocol

Benchmark Overhaul
```

---

# 43. 关键测试

## MCP

```text
test_agent_mcp_tools_list_contains_reads

test_agent_mcp_tools_list_contains_proposals

test_agent_mcp_never_exposes_real_write_tools

test_runtime_mcp_exposes_write_tools
```

---

## Proposal

```text
test_proposal_tool_has_zero_platform_side_effect

test_multiple_proposals_same_round_are_rejected

test_proposal_must_be_single_tool_call
```

---

## HITL

```text
test_reject_never_mutates

test_approve_executes_exact_frozen_action

test_wrong_fingerprint_cannot_execute

test_edit_invalidates_old_approval
```

---

## Precondition

```text
test_stale_approval_is_blocked
```

---

## WRITE

```text
test_write_executes_at_most_once_per_approval

test_unknown_outcome_does_not_retry_write

test_unknown_outcome_reconciles_by_read
```

---

## Verification

```text
test_api_success_but_state_not_changed_is_not_success

test_final_guard_blocks_false_success
```

---

## Pipeline

```text
test_prepare_task_spec_has_no_side_effect

test_submit_requires_prepared_artifact

test_submit_executes_exact_reviewed_yaml
```

---

# 44. 项目面试主线

不要只说：

```text
我用了 LangGraph + MCP + RAG
```

应该说：

> 这个项目解决的是自动驾驶训练平台的自然语言 DataOps 操作问题。我使用 Single-Agent ReAct 让模型根据实时 Task、GPU、Queue 和 RAG Observation 动态选择后续 Tool；平台能力通过标准 MCP 接入，MCP 的 Tool Schema 映射到模型 Native Function Calling。READ Tool 可以由 Agent 自主执行，但真实平台 WRITE Tool 不直接暴露给模型，而是通过对应的无副作用 Proposal Tool生成结构化修改建议。Proposal 经 HITL Review 并冻结参数后，由 deterministic WriteService 通过 Runtime MCP 调用真实 WRITE Tool，执行前检查平台 Precondition，执行后重新读取真实平台状态进行 Post-write Verification。

最终概括：

> **LangGraph 管流程，MCP 管工具接入，LLM 管语义决策，Runtime 管确定性执行。**

---

# 45. 简历方向（实现完成并 Code Review 后使用）

## Agent + MCP

基于 LangGraph 构建 Single-Agent ReAct Loop，将原有自定义 JSON-RPC Tool Gateway 重构为标准 MCP Client/Server，通过 Streamable HTTP 与 Tool Discovery 统一接入任务状态、GPU/Queue、故障诊断及 RAG 等平台能力，并将 MCP Tool Schema 映射至模型 Native Function Calling。

## Tool 权限隔离与 HITL

基于共享 Tool Registry 构建 Agent/Runtime 两类 MCP Capability Profile，模型仅可发现 READ、Prepare 及无副作用 Proposal Tool，真实 Mutation Tool 仅供 Runtime 使用；对优先级调整、任务启停、删除及 Pipeline 提交生成结构化 Proposal，并通过 Artifact / Before-After Review 与 fingerprint 绑定最终审批参数。

## WRITE Reliability

封装 deterministic WriteService，审批后通过 Precondition Recheck 防止状态漂移，结合 idempotency key、single mutation attempt 和 UNKNOWN_OUTCOME read reconciliation 避免重复副作用；执行后重新读取平台真实状态进行 Post-write Verification，仅允许 verified WriteResult 对外报告成功。

## Pipeline / RAG

将自然语言 Pipeline 请求抽取为 TaskDraft，并结合平台 deterministic defaults 与 Schema Validation 生成 TaskSpec/YAML Artifact；RAG 作为普通 MCP READ Tool，采用 BM25 + Dense Embedding Hybrid Retrieval，为任务诊断、队列策略及运行手册提供知识 Observation。

---

# 46. 最终架构宪法

以后所有 V3.1 设计与代码 Review 都应以这段为准：

> **AutoDriveDataOpsAgent V3.1 是一个基于 LangGraph 的 Single-Agent Guarded ReAct 系统。平台 READ、RAG、Prepare、Proposal 与 WRITE 能力通过标准 MCP 统一接入，并通过 Agent/Runtime Capability Profile 隔离模型可见能力和真实副作用权限；Agent 使用 Native Function Calling 自主调用无副作用 Tool 感知环境并生成结构化 Proposal，Runtime 基于实时平台状态构造并冻结 PendingAction，经 HITL 审批后由 deterministic WriteService 完成 Precondition Recheck、幂等控制、Runtime MCP Mutation 和 Post-write Verification，最终只允许基于 verified result 报告写操作成功。**

最终四句话：

> **LLM 管语义，Runtime 管确定性。**

> **LangGraph 管流程，MCP 管工具接入。**

> **Proposal ≠ Execution。**

> **API success ≠ business success。**
