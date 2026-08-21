# AutoDriveDataOpsAgent A+ 最终开发总体规划

## 1. 文档目的

本文档定义 AutoDriveDataOpsAgent 从当前 V1.4.3 继续演进到 **A+ 最终版本**的总体目标、架构边界、版本路线、验收标准和明确不做的内容。

A+ 不是完整无人值守的 Autonomous Operator，也不只是所有操作都依赖人工确认的传统 Copilot。

A+ 的目标定位为：

> **Strong DataOps Copilot + Bounded Autonomous Slice**

即：

- Agent 可以自主理解目标；
- 自主查询平台状态；
- 自主选择下一步 Tool；
- 自主进行多轮 Observe → Reason → Act；
- 自主聚合实时 Evidence 与平台 Knowledge；
- 自主完成故障诊断和操作方案制定；
- 高风险 Write Action 继续严格 HITL；
- 选择一种边界清晰的低风险恢复操作，在 deterministic Policy 允许时自动执行；
- 所有实际操作必须经过 Precondition；
- 所有操作完成后必须重新 Observe 并 Verify；
- 所有自治行为受到 Action Budget、Loop Budget 和 Audit 约束。

A+ 完成以后，项目应从：

```text
LLM + MCP Tool Calling
```

进一步演进为：

```text
Bounded Autonomous DataOps Agent
```

---

# 2. 当前基线

当前开发基线：

```text
V1.4.3
Agent Routing & Tool Semantics
```

当前已经具备：

```text
Platform Core
GPU Simulator
Mock Stage
Airflow Multi-task
Docker Lifecycle
Priority Queue
Soft Preemption
Recovery
GPU Reservation

Platform MCP
Read-only Tools
Write Tools

Natural Language Understanding
Task Planning
RAG
search_knowledge as Agent Tool
Evidence Routing
Tool Semantics
HITL
Precondition
Action Verification
Tracing
Audit
Evaluation
Qwen / Gemini / OpenAI Provider
```

V1.4.3 已解决：

```text
Static Knowledge
vs
Live Runtime State
vs
Task Diagnosis
vs
Task Planning
vs
Write Operation
vs
No-tool
```

之间的主要 Routing Contract。

真实 qwen-plus 已达到：

```text
Intent Accuracy = 1.0
Tool Precision = 1.0
Tool Recall = 1.0
Tool F1 = 1.0
Forbidden Tool Rate = 0
No-tool Accuracy = 1.0
```

当前最大的架构限制不再是：

```text
“模型会不会选对 Tool？”
```

而是：

```text
“模型能否根据一次 Tool Observation，
动态决定下一步还需要什么 Evidence？”
```

当前 workflow 仍接近：

```text
User
↓
Plan all tools
↓
Execute tools
↓
Synthesize
```

A+ 的核心任务就是突破这一限制。

---

# 3. A+ 最终项目定位

最终项目定位：

> **面向自动驾驶离线数据处理平台的 Bounded Autonomous DataOps Agent。通过 MCP 将 Airflow、Docker、GPU Reservation、Priority Queue、Soft Preemption、Recovery 和平台知识等能力抽象为领域工具，由 Stateful Agent 根据用户目标进行多步观察、动态工具决策、证据驱动诊断和操作规划；高风险操作通过 HITL 执行，满足 deterministic Policy 的低风险恢复操作允许有限自治，并通过 Precondition、Execution Budget、Action Verification 和 Goal Verification 保证执行安全。**

项目不定位为：

```text
Airflow Chatbot
LLM + Airflow REST API
RAG Demo
MCP Demo
Multi-Agent Demo
完全无人值守生产运维系统
```

---

# 4. 最终设计原则

## 4.1 Platform Core 永远是确定性执行面

以下能力继续由代码和平台规则负责：

```text
Airflow execution
Task lifecycle
Queue ordering
GPU allocation
GPU Reservation
exclusive/shared rule
Soft Preemption
Checkpoint validity
Recovery start point
Docker lifecycle
TaskSpec validation
Write precondition
Action verification
```

LLM 不得自行决定这些规则。

例如：

错误：

```text
LLM:
GPU0 还有 20GB，
Segment 需要 16GB，
所以我判断可以运行。
```

正确：

```text
Agent:
我需要判断 Segment 是否能获得 GPU。

↓ API / Tool

GPUService:
按照真实 memory + reservation + exclusive policy
进行 deterministic 判断。

↓

Agent:
解释结果。
```

---

## 4.2 Agent 是智能控制面

Agent 负责：

```text
理解用户目标
判断需要什么 Evidence
选择下一步 Tool
分析 Tool Observation
判断 Evidence 是否充分
查询平台知识
形成 Diagnosis
生成 Action Plan
判断是否需要继续 Observe
解释平台状态
决定是否提交 Proposed Action
```

核心职责：

```text
What should I inspect?
What evidence is missing?
What should I do next?
Is the goal already satisfied?
How should I explain the result?
```

---

## 4.3 Evidence First

Agent 不应主要依赖模型记忆回答平台事实。

事实来源分为：

```text
Operational Evidence
+
Knowledge Evidence
```

Operational Evidence：

```text
Task
Queue
GPU
Container
Airflow
Checkpoint
Logs
Platform Health
```

必须来自 MCP / Platform Core。

Knowledge Evidence：

```text
Architecture
GPU Reservation rules
Soft Preemption
Recovery
Task lifecycle
Runbook
Failure handling
Historical incidents
```

必须来自：

```text
search_knowledge
```

原则：

```text
Current state
→ Operational Tool

Platform mechanism
→ search_knowledge

Diagnosis
→ Operational Evidence first
   + Knowledge when needed
```

---

# 5. A+ 最重要的架构升级：Adaptive Agent Loop

## 5.1 当前模式

当前主要流程：

```text
User
↓
Planner
↓
一次性生成 Tool Calls
↓
Execute
↓
Answer
```

问题：

Planner 在执行 Tool 之前并不知道 Observation。

因此只能提前猜：

```text
未来可能需要哪些 Tool。
```

复杂 Diagnosis 很容易出现：

```text
调用过多 Tool
```

或者：

```text
第一次 Observation 出现新线索，
但 Agent 已经没有机会改变计划。
```

---

## 5.2 A+ 模式

目标流程：

```text
User Goal
   ↓
Understand
   ↓
Decide Next Step
   ↓
Tool
   ↓
Observe
   ↓
Reason
   ↓
Evidence Enough?
   │
   ├── No
   │    ↓
   │ Decide Next Tool
   │    ↓
   │ Tool
   │    ↓
   │ Observe
   │    ↓
   │ ...
   │
   └── Yes
        ↓
     Answer / Action Plan
```

Agent 不再一次性决定所有 Tool。

而是：

> 每得到新的 Evidence 后，重新决定下一步。

---

# 6. Adaptive Loop 示例

用户：

```text
release_demo 为什么一直没跑？
```

第一步：

```text
diagnose_task(release_demo)
```

Observation：

```text
task_state=active
current_stage=segment
stage_state=waiting_gpu
```

Agent 判断：

```text
问题已经缩小到 GPU resource。
```

下一步：

```text
get_gpu_pool()
```

Observation：

```text
GPU0:
OD exclusive reservation

GPU1:
free_memory=18GB

Segment:
required=24GB
exclusive=true
```

Agent 判断：

```text
实时原因已经明确。
```

如果用户只问：

```text
为什么没跑？
```

可以结束。

如果用户问：

```text
为什么这种情况下 Segment 不能共享 GPU？
```

Agent 再调用：

```text
search_knowledge(
    "Segment exclusive GPU reservation rules"
)
```

最后形成：

```text
Operational Evidence
+
Knowledge Evidence
+
Diagnosis
```

---

# 7. Adaptive Loop 必须是有界的

不能让 Agent：

```text
Tool
→ Tool
→ Tool
→ Tool
→ ...
```

无限执行。

必须引入：

```text
Loop Budget
```

建议至少包含：

```text
max_steps
max_tool_calls
max_same_tool_calls
max_reobserve_count
max_consecutive_failures
```

例如：

```text
max_steps = 8
max_tool_calls = 6
max_same_tool_calls = 2
```

具体默认值以后通过 Evaluation 调整。

超过 Budget：

```text
STOP
↓
输出当前 Evidence
↓
说明缺失信息
↓
Escalate / Ask user
```

禁止无限 Agent Loop。

---

# 8. Evidence Sufficiency

Adaptive Loop 不能只靠模型自由决定什么时候结束。

需要逐步建立：

```text
Evidence Sufficiency Contract
```

例如 task diagnosis：

最低需要：

```text
Task identity
Task state
Relevant execution evidence
```

GPU diagnosis：

最低需要：

```text
Live GPU state
Relevant reservation evidence
```

Knowledge question：

最低需要：

```text
Knowledge Evidence
```

Write planning：

最低需要：

```text
Target identity
Current state
Impact evidence
```

Agent 可以判断是否需要继续获取 Evidence。

但 deterministic workflow 必须保证：

```text
关键 evidence requirement 没满足
→ 不允许进入 Write execution。
```

---

# 9. 从 Action Verification 升级到 Goal Verification

当前已有：

```text
Action Verification
```

例如：

```text
set priority
↓
重新查询 queue
↓
确认 priority 已变化
```

A+ 需要进一步增加：

```text
Goal Verification
```

区别：

Action Verification：

```text
操作是否成功？
```

Goal Verification：

```text
用户目标是否实现？
```

例如用户说：

```text
帮我恢复 release_demo。
```

Agent：

```text
resume_task
```

Tool 成功不代表用户目标完成。

必须继续观察：

```text
task 是否离开 interrupted state
DagRun 是否创建
Stage 是否继续执行
checkpoint 是否正确
任务是否再次失败
```

最终：

```text
goal = task resumed successfully
```

才能报告成功。

---

# 10. Write Path 最终结构

A+ 最终 Write Path：

```text
User Goal
↓
Observe
↓
Reason
↓
Build Proposed Action
↓
Impact Analysis
↓
Risk / Policy
↓
┌───────────────────────────────┐
│                               │
AUTO                         APPROVAL
│                               │
│                         HITL Approval
│                               │
└──────────────┬────────────────┘
               ↓
        Frozen Arguments
               ↓
          Precondition
               ↓
            Execute
               ↓
         Observe Again
               ↓
      Action Verification
               ↓
       Goal Verification
               ↓
       Success / Escalate
```

---

# 11. A+ 的自治范围

A+ 不追求：

```text
所有 Write Tool 自动执行。
```

最终只有：

```text
一个明确、低风险、确定性规则充分的 Autonomous Slice。
```

首选目标：

```text
Policy-gated safe resume
```

即：

> 对满足确定性 Recovery / Checkpoint / Task State 条件的可恢复任务，允许 Agent 自动调用 `resume_task`。

---

# 12. 为什么选择 resume_task

相比：

```text
submit_task
set_task_priority
stop_task
delete_task
```

安全恢复类操作更适合成为 A+ 的第一类自治行为。

原因：

```text
影响范围相对容易限制
目标明确
可验证
已有 Recovery / Checkpoint 机制
可以设置严格 Preconditions
可以限制执行次数
失败后容易 Escalate
```

以下行为在 A+ 阶段保持 HITL：

```text
submit_task
set_task_priority
stop_task
delete_task
```

特别是：

```text
delete_task
```

始终属于高风险操作。

---

# 13. Safe Resume 的 Autonomous Policy

不能由 LLM 判断：

```text
“我觉得这个 resume 风险比较低。”
```

必须通过 deterministic Policy。

建议 Auto Resume Eligibility 至少考虑：

```text
action_type == resume_task

target task exists

task state explicitly recoverable

valid checkpoint exists

checkpoint passed Stage Validation

no conflicting pending write action

no stale approval/action claim

current state matches observed state

no cross-task destructive impact

resume does not require priority mutation

resume budget not exhausted

previous autonomous attempt did not already fail

verification strategy exists
```

只有全部满足：

```text
AUTO_ALLOWED
```

否则：

```text
HITL_REQUIRED
```

或：

```text
DENY
```

---

# 14. Risk Classification

A+ 引入轻量 Risk Classification。

不需要构建通用风险 AI 模型。

风险由 deterministic metadata / Policy 得出。

建议：

```text
NONE
LOW
MEDIUM
HIGH
CRITICAL
```

初始 Policy：

```text
Read-only
→ NONE

search_knowledge
→ NONE

safe resume under strict policy
→ LOW

submit
→ MEDIUM

set priority
→ MEDIUM/HIGH

stop
→ HIGH

delete
→ CRITICAL
```

A+ 自动执行只允许：

```text
LOW
```

---

# 15. Autonomy Budget

即使 Action 属于 LOW risk，也不能无限执行。

引入：

```text
Autonomy Budget
```

至少记录：

```text
max_auto_actions_per_session
max_auto_actions_per_task
max_resume_attempts
max_impacted_tasks
max_consecutive_failures
```

例如目标设计：

```text
max_auto_actions_per_session = 3
max_auto_actions_per_task = 1
max_resume_attempts = 1
max_impacted_tasks = 1
```

超过 Budget：

```text
AUTO
↓
HITL / ESCALATE
```

---

# 16. Autonomous Failure Handling

自动 resume 后可能：

```text
成功
失败
超时
Precondition Failed
Verification Failed
再次进入异常
```

必须规定：

### Success

```text
Goal Verification PASS
→ 完成
```

### Precondition Failed

说明 Observe 后状态已经变化：

```text
不得继续执行旧 Action
↓
重新 Observe
↓
重新 Reason
```

### Verification Failed

```text
不得报告成功
↓
重新 Diagnosis
```

### Auto Action Failed

```text
不得无限重试
↓
consume budget
↓
Escalate to HITL
```

---

# 17. A+ Tracing

现有 tracing 继续保留，并扩展 adaptive / autonomy 字段。

至少记录：

```text
trace_id
session_id
user_request
goal
intent

agent_step
step_type

tool_name
tool_args
tool_result

observation

knowledge_sources

reasoning_decision_summary

evidence_sufficient

proposed_action

risk_level
policy_decision

approval

autonomy_budget_before
autonomy_budget_after

precondition

action_result
action_verification
goal_verification

termination_reason

latency
token_usage
final_result
```

注意：

不记录模型私有 Chain-of-Thought。

记录的是：

```text
可审计的决策摘要
结构化状态
Evidence
Tool
Policy
Verification
```

---

# 18. Evaluation 总体升级

A+ Evaluation 不再只测试：

```text
Intent
Tool Selection
```

最终需要五层 Eval。

---

## 18.1 Routing Eval

测试：

```text
用户需求
→ intent
→ evidence class
→ first tool
```

包括：

```text
Static Knowledge
Live Task
Live GPU
Diagnosis
Planning
Write
No-tool
```

---

## 18.2 Adaptive Planning Eval

测试：

```text
Observation 1
↓
是否选择正确 Tool 2
```

例如：

```text
diagnose_task
→ waiting_gpu

Expected next:
get_gpu_pool
```

而不是：

```text
search_knowledge
```

---

## 18.3 Diagnosis Eval

Simulator 构造确定性故障：

```text
GPU insufficient
exclusive reservation blocked
stale reservation
container missing
stage failed
validate failed
checkpoint invalid
task queued
task draining
```

Expected：

```text
correct root cause
correct supporting evidence
```

---

## 18.4 Safety Eval

必须保证：

```text
Forbidden direct write rate = 0
```

并测试：

```text
delete without approval
stop without approval
priority mutation without approval
unsafe resume auto execution
stale precondition
budget exceeded
```

全部必须被拒绝或升级 HITL。

---

## 18.5 Autonomous Execution Eval

重点测试：

```text
safe resume
```

场景：

### Case A

符合全部 Policy：

```text
Expected:
AUTO
→ resume
→ verify
→ PASS
```

### Case B

checkpoint invalid：

```text
Expected:
NO AUTO
```

### Case C

state changed after planning：

```text
Expected:
PRECONDITION_FAILED
→ no write
```

### Case D

resume Tool success but state wrong：

```text
Expected:
Verification Failed
→ not report success
```

### Case E

first autonomous resume fails：

```text
Expected:
no infinite retry
→ Escalate
```

---

# 19. Evaluation 数据集设计原则

正式 Eval 必须逐渐与 Prompt examples 分离。

禁止：

```text
Prompt 中直接出现的例句
=
Formal Eval Case
```

需要：

```text
Holdout
Paraphrase
Adversarial overlap
Multi-turn
State transition
```

特别测试：

```text
关键词相同但 evidence class 不同
```

例如：

```text
“GPU Reservation 是什么？”
→ knowledge

“GPU0 现在有什么 Reservation？”
→ live GPU

“release_demo 为什么被 Reservation 卡住？”
→ task diagnosis

“结合当前 GPU 状态和 Reservation 规则解释”
→ operational + knowledge
```

---

# 20. Argument Evaluation

Tool 参数不能全部使用 exact string comparison。

至少区分：

```text
EXACT
SEMANTIC
OPTIONAL
RANGE
SUBSET
```

例如：

```text
task_name
→ EXACT

priority
→ EXACT

dataset
→ EXACT

search_knowledge.query
→ SEMANTIC

top_k
→ RANGE / OPTIONAL
```

不要为了 Argument Score 强迫模型复制用户原句。

---

# 21. A+ 版本路线

---

## V1.4.4 — Evaluation Hardening

### 目标

冻结 V1.4.3 production routing。

优先验证泛化能力，而不是继续调 Prompt。

### 工作

```text
Holdout Routing Dataset
Adversarial Cases
Semantic Argument Contract
Evaluation Contract Cleanup
Cross-provider compatibility
```

### 原则

只有 holdout 真正发现系统性 production 问题时，才允许修改 Prompt。

### 验收

```text
Routing holdout stable
Forbidden write rate = 0
No-tool stable
Historical V1.4.x regression pass
```

---

## V1.5.0 — Adaptive Agent Loop

### 目标

从：

```text
Plan All Tools
```

升级为：

```text
Decide Next Tool
→ Observe
→ Decide Again
```

### 增加

```text
Agent Step
Observation State
Loop Controller
Step Budget
Tool Budget
Termination Reason
Evidence Sufficiency
```

### 验收

典型 multi-step diagnosis 能根据 Observation 动态选择下一 Tool。

不得：

```text
提前写死所有 Tool sequence
```

不得无限 loop。

Write safety boundary 不变。

---

## V1.6.0 — Long-horizon Diagnosis & Goal Completion

### 目标

让 Agent 不只回答：

```text
发生了什么？
```

还能够完成：

```text
帮我把这个问题处理到可交付状态。
```

### 增加

```text
Goal Model
Diagnosis State
Plan Revision
Re-observation
Goal Verification
Escalation
```

### 重点场景

```text
Task not running
GPU blocked
Task interrupted
Recovery
Container failure
Validation failure
```

### 验收

Agent 可以：

```text
Observe
→ Diagnose
→ Plan
→ Act/HITL
→ Observe
→ Revise
→ Verify Goal
```

---

## V1.7.0 — Bounded Autonomy

### 目标

正式加入 A+ 的 autonomous slice。

### 自动操作范围

第一阶段仅：

```text
policy-gated safe resume_task
```

其他 Write：

```text
submit
priority
stop
delete
```

继续 HITL。

### 增加

```text
Risk Classification
Autonomy Eligibility
Autonomy Budget
Auto/HITL/Deny Policy
Failure Escalation
```

### 验收

所有 Auto Resume：

```text
100% 经过 deterministic Policy
100% 经过 Precondition
100% 经过 Verification
```

任何不满足条件的 Action：

```text
不得 AUTO。
```

---

## V1.8.0 — A+ Final Hardening

### 目标

形成秋招可交付的最终稳定版本。

### 工作

```text
Regression
Scenario Evaluation
Failure Injection
Tracing / Audit Review
Documentation
Architecture Diagram
Final Benchmark
Release Report
```

最终输出至少包含：

```text
Agent Capability Matrix
Safety Matrix
Autonomous Policy Matrix
Evaluation Report
Architecture
Known Limitations
```

---

# 22. 最终 A+ 架构

```text
                         User Goal
                             │
                             ▼
                     DataOps Agent
                             │
                  Understand / Route
                             │
                             ▼
                    Adaptive Agent Loop
                             │
               ┌─────────────┴──────────────┐
               │                            │
        Operational Evidence         Knowledge Evidence
               │                            │
            MCP Tools                 search_knowledge
               │                            │
               └─────────────┬──────────────┘
                             │
                           Reason
                             │
                  Evidence Sufficient?
                       │           │
                      No          Yes
                       │           │
                Next Tool       Answer
                                   │
                            Proposed Action?
                              │          │
                             No         Yes
                              │          │
                           Finish      Policy
                                         │
                         ┌───────────────┼───────────────┐
                         │               │               │
                       AUTO            HITL            DENY
                         │               │
                         └───────┬───────┘
                                 │
                          Precondition
                                 │
                              Execute
                                 │
                          Observe Again
                                 │
                       Action Verification
                                 │
                         Goal Verification
                                 │
                         Success / Escalate
```

---

# 23. LLM 与 Deterministic Code 最终职责

## LLM

负责：

```text
Intent understanding
Evidence routing
Next-tool decision
Diagnosis reasoning
Knowledge synthesis
Plan generation
Explanation
```

## Deterministic Code

负责：

```text
TaskSpec legality
GPU allocation legality
Queue ordering
Recovery legality
Checkpoint validation
Risk policy
Autonomy eligibility
Action budget
Approval state
Frozen arguments
Preconditions
Tool execution
Action verification
Goal invariants
```

核心原则：

> LLM 决定“下一步应该了解什么、建议做什么”，确定性系统决定“这个动作是否允许以及是否真的成功”。

---

# 24. 明确不做

A+ 阶段不做：

```text
Multi-Agent
Kubernetes migration
Web Dashboard
Full autonomous operations
Automatic delete
Automatic stop
Automatic priority mutation
Automatic task submission
LLM-based risk approval
LLM-based GPU scheduling
LLM direct shell
LLM direct Docker
LLM direct DB mutation
Unlimited agent loop
Unlimited autonomous retry
```

不为了“Agent 味道”增加没有实际价值的复杂度。

---

# 25. A+ 最终成功标准

A+ 完成必须同时满足：

## Agent Capability

```text
[ ] Natural Language Understanding
[ ] Tool Routing
[ ] RAG as Agent Tool
[ ] Adaptive Tool Loop
[ ] Evidence-driven Diagnosis
[ ] Multi-step Planning
[ ] Task Planning
[ ] Goal Tracking
[ ] Plan Revision
[ ] Goal Verification
```

## Safety

```text
[ ] No direct write
[ ] HITL for medium/high risk
[ ] Deterministic Policy
[ ] Frozen arguments
[ ] Precondition
[ ] Action Verification
[ ] Goal Verification
[ ] Autonomy Budget
[ ] Loop Budget
[ ] Forbidden Write Rate = 0
```

## Autonomous Slice

```text
[ ] Only explicitly allowed low-risk action can AUTO
[ ] AUTO eligibility deterministic
[ ] Invalid checkpoint cannot auto resume
[ ] Stale state cannot execute
[ ] Failed autonomous action cannot loop indefinitely
[ ] Budget exhaustion escalates
[ ] Every autonomous action is auditable
```

## Engineering

```text
[ ] compileall PASS
[ ] git diff --check PASS
[ ] doctor --strict PASS
[ ] E2E PASS
[ ] relevant regression PASS
[ ] dependency-light PASS except documented environment-only issues
```

## Evaluation

至少能够真实报告：

```text
Intent Accuracy
Tool Precision
Tool Recall
Tool F1
Argument Correctness
Routing Holdout Accuracy
Adaptive Next-step Accuracy
Diagnosis Accuracy
Scenario Completion Rate
Forbidden Write Tool Rate
Unsafe Auto Action Rate
Verification Accuracy
Goal Completion Accuracy
No-tool Accuracy
Average Agent Steps
Tool Calls per Task
Latency
Token Usage
```

安全指标不能依赖平均分掩盖失败。

例如：

```text
Unsafe Auto Action Rate
```

目标必须为：

```text
0
```

---

# 26. A+ 最终典型场景

用户：

```text
看看 release_demo 为什么没继续跑，如果只是安全的恢复问题就帮我处理。
```

Agent：

```text
1. diagnose_task(release_demo)

2. Observation:
   task interrupted
   checkpoint valid
   previous stage validated

3. Agent 判断需要确认恢复条件。

4. 获取必要实时 Evidence。

5. Proposed Action:
   resume_task(release_demo)

6. Deterministic Policy:
   LOW RISK
   checkpoint valid
   no cross-task impact
   budget available

7. AUTO_ALLOWED

8. Precondition PASS

9. resume_task

10. Observe Again

11. Task entered running/recovery state

12. Goal Verification PASS
```

最终回答：

```text
release_demo 在已验证 checkpoint 后中断，
符合安全恢复条件。

我已按低风险恢复策略自动执行 resume。
恢复后重新检查，任务已经从 checkpoint 继续运行，
当前状态正常。

本次没有修改优先级、停止其他任务或删除任何资源。
```

---

另一个用户：

```text
release_demo 很急，直接把优先级改到最高。
```

Agent：

```text
Observe
↓
Impact Analysis
↓
发现当前 active task 会进入 draining
↓
Risk = MEDIUM/HIGH
↓
AUTO_NOT_ALLOWED
↓
HITL
```

最终：

```text
可以修改，但该操作会影响当前运行中的其他业务任务，
因此不会自动执行，需要确认。
```

这体现 A+ 的核心：

> **Agent 有自治能力，但自治能力被限制在明确、安全、可验证的边界内。**

---

# 27. 秋招最终项目故事

A+ 完成后，项目可以形成完整的技术演进故事：

```text
传统离线处理平台
↓
Platform Core 模块化
↓
GPU Simulator / Mock Runtime
↓
MCP Domain Tools
↓
Read-only Agent
↓
RAG
↓
Natural Language Task Planning
↓
Write Agent + HITL
↓
Precondition + Verification
↓
Tracing + Evaluation
↓
RAG as Agent Tool
↓
Evidence Routing
↓
Adaptive Agent Loop
↓
Long-horizon Diagnosis
↓
Risk-aware Bounded Autonomy
```

最终不是：

```text
“我接了一个大模型。”
```

而是：

> **我把一个已有复杂调度平台逐步演进成一个具备动态工具决策、证据驱动诊断、安全写治理和有限自治能力的 DataOps Agent。**

---

# 28. A+ 完成后的停止点

当 V1.8.0 A+ Final 达到验收标准后：

优先：

```text
冻结主架构
补充 Evaluation
完善 README
整理项目故事
准备简历与面试
```

而不是立刻继续增加功能。

只有在：

```text
A+ 已稳定
测试完整
文档完整
Eval 可解释
秋招材料准备完成
仍有明显时间冗余
```

的情况下，才进入完整 B 方案。

---

# 29. B 的后续方向

B 不属于当前必做范围。

可能包括：

```text
multiple autonomous write actions
broader risk matrix
cross-task remediation
automatic low-risk retry
automatic recovery orchestration
dynamic autonomy budget
scheduled / event-driven monitoring
proactive incident handling
```

进入 B 前必须保证：

```text
A+ safety invariants
```

不被破坏。

---

# 30. 最终原则

整个 A+ 开发过程中始终遵守：

> 用户负责描述目标。

> Agent 负责决定应该观察什么、查询什么、下一步做什么。

> Platform Core 负责保证平台规则确定执行。

> Policy 负责决定 Agent 是否允许行动。

> Precondition 负责保证行动时世界仍然与计划一致。

> Verification 负责证明操作真的成功。

> Budget 负责保证 Agent 不会无限行动。

> Audit 负责保证整个过程可以复盘。

最终形成：

```text
Observe
↓
Reason
↓
Decide
↓
Act
↓
Observe
↓
Verify
↓
Complete or Escalate
```

这就是 AutoDriveDataOpsAgent A+ 的最终目标。
