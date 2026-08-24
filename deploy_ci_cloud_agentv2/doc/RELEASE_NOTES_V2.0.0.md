# AutoDriveDataOpsAgent V2.0.0

## Release profile

`v2.0.0` is the first release of the V2 single-node simulated deployment.
It combines a real Alibaba Bailian/Qwen Agent with the real V2 LangGraph,
Runtime, safety transactions, persistence, gateway, RAG, and backing-service
deployment. The managed stage and GPU environments are intentionally
`mock`/`simulated`.

```text
PLATFORM_STAGE_RUNTIME=mock
PLATFORM_GPU_RUNTIME=simulated
AUTODRIVE_GATEWAY_BACKEND=in_process
AUTODRIVE_PROVIDER=qwen
AUTODRIVE_MODEL=qwen3.7-plus-2026-05-26
AUTODRIVE_PROVIDER_STRUCTURED_MODE=json_schema
AUTODRIVE_PROVIDER_THINKING_MODE=auto
PLATFORM_RAG_EMBED_PROVIDER=local
```

## Included capabilities

- Real Qwen semantic Agent with strict JSON-Schema `AgentDecision` output.
- Strict local parsing and `DecisionIngress` admission after model output.
- Real LangGraph single-loop READ/WRITE/FINAL control flow.
- Five canonical platform READ tools and protected WRITE tools.
- Bounded parallel READ batches.
- Human-approved WRITE transactions with precondition revalidation.
- Single-use `ExecutionClaim`, TOCTOU protection, and mutation accounting.
- `OUTCOME_UNKNOWN` and READ-only reconciliation.
- Independent action/operational verification and CompletionGate.
- Frozen BM25/local-hashing RAG with optional Qwen dense embeddings.
- Immutable 30-case RAG evaluation asset and packaged knowledge assets.
- Durable SQLite checkpoints, approvals, events, and execution claims.
- Clean restart and release artifact validation.

## Safety and scope

The release performs no production WRITE. Approval, transaction, claim,
verification, and reconciliation boundaries remain Runtime-owned. Physical
multi-GPU hardware, a multi-node cluster, and non-mock AutoDrive are not
release requirements:

```text
NON_MOCK_AUTODRIVE=OUT_OF_SCOPE
PHYSICAL_MULTI_GPU=OUT_OF_SCOPE
```

The release therefore claims a single-node simulated deployment, not a
physical cluster or external AutoDrive deployment.

## Provider and RAG notes

The validated Agent model is `qwen3.7-plus-2026-05-26`. In automatic thinking
policy, strict JSON-Schema Agent requests send `enable_thinking=false` to
control latency. `DecisionIngress` remains mandatory. The default RAG mode
is offline local retrieval; Qwen dense embedding is an optional validated
mode and no reranker is included (`RERANKER_NOT_REQUIRED`).

## Operations

Install the wheel into the supported runtime, load `DASHSCOPE_API_KEY` from
the external mode-600 credential file, start the backing services, and run
the canonical V2 gateway launcher. See:

- `SINGLE_NODE_SIMULATED_DEPLOYMENT.md`
- `OPERATIONS_RUNBOOK.md`
- `DEPLOYMENT.md`

The v2.0.0 release artifact is accompanied by a SHA-256 checksum and a
machine-readable release manifest. Secrets, databases, logs, and temporary
state are not release artifacts.

## Known intentional limitations

- Stage execution and GPU inventory/workload behavior are simulated.
- The deployment is single-node and single-instance for SQLite state.
- Non-mock AutoDrive and physical multi-GPU validation are out of scope.
- Qwen availability depends on Alibaba service, network, account, and quota.
- Local RAG is the default; external dense embedding is optional.
- No reranker is included in this release.
