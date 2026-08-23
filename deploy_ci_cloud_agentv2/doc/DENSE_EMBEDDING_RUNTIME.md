# Dense Embedding Runtime Integration

V2 platform RAG keeps BM25 as its lexical signal. Dense retrieval is optional
infrastructure below the platform tool boundary; it is not an Agent, approval,
verification, or completion authority. V2 deliberately has no reranker.

## Modes

`PLATFORM_RAG_EMBED_PROVIDER` controls the vector branch constructed by
`platform_backend.runtime.build_platform_facade()`.

| Value | Vector branch | Network required |
|---|---|---|
| unset, `local`, or legacy `hash` | deterministic 384-dimensional feature hashing | No |
| `qwen` | `QwenEmbeddingProvider` dense cosine | Yes |
| `gemini` | `GeminiEmbeddingProvider` dense cosine | Yes |

The default is `local`, so normal startup stays offline and does not require a
secret. Explicit external modes fail during construction when required
configuration is absent; they never silently mix a dense index with hashing.

## Configuration

```bash
# Qwen
export PLATFORM_RAG_EMBED_PROVIDER=qwen
export PLATFORM_RAG_EMBED_MODEL=qwen3.7-text-embedding
export PLATFORM_RAG_EMBED_DIM=1024
export PLATFORM_RAG_EMBED_BATCH_SIZE=20
export DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/api/v1
# Inject DASHSCOPE_API_KEY through the process environment only.

# Gemini
export PLATFORM_RAG_EMBED_PROVIDER=gemini
export PLATFORM_RAG_EMBED_MODEL=gemini-embedding-2
export PLATFORM_RAG_EMBED_DIM=768
export PLATFORM_RAG_EMBED_BATCH_SIZE=32
# Inject GEMINI_API_KEY or GOOGLE_API_KEY through the process environment only.
```

The canonical wheel declares the `requests` and `google-genai` dependencies
used by the existing provider adapters. Secrets are never written to sidecars,
logs, prompts, or documentation.

## Dense index lifecycle

The lexical index defaults to
`<AUTODRIVE_RUNTIME_ROOT>/state/v2_knowledge/index.json`; dense mode writes an
atomic resumable `<index>.embeddings.json` sidecar. Metadata binds schema,
source fingerprint, provider, model, dimension, expected chunk count, content
hashes, and completion. Provider/model/dimension mismatches cannot reuse
vectors from another embedding space. Unchanged chunks are reused; changed/new
chunks are recomputed; deleted chunks are absent from the next sidecar.

Document and query responses are validated before use: item identity/count,
configured dimension, and finite values are required. Qwen sends documents as
`text_type=document` and searches as `text_type=query`; Gemini retains its
existing distinct document/query formatting.

The existing hybrid formula is unchanged: lexical weight `0.65`, vector weight
`0.35`, then the heading/domain bonus. Local mode uses feature-hashing cosine;
dense mode uses dense cosine.

## Validation evidence

The five-case `retrieval_golden.json` remains a fast smoke set; it is not the
final V1.1 benchmark. The canonical V1.1 evaluation asset is the unchanged
30-case JSONL at:

```text
platform_backend/knowledge/eval/v1_1/rag_retrieval.jsonl
```

Its SHA-256 is
`7cb32e25fbb35a62274732558ed00f42aa98f20c871c7281127247efcb19f7ed`.
The dependency-light evaluator validates the 30-case schema and source/section
references, then calls the production `KnowledgeService` retriever for both
modes. Per-case artifacts are written outside the source tree under the
runtime evaluation state directory; they contain IDs/ranks, not secrets or
document vectors.

On 2026-08-24, real Qwen `qwen3.7-text-embedding` document and query smokes
returned normalized finite 1024-dimensional vectors. The validated dense
sidecar contains 443/443 vectors for 30 documents and was reused for the
30-case query evaluation; no document vectors were regenerated.

The canonical 30-case A/B result was:

| Metric | Local BM25 + hash | Qwen dense hybrid | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.6667 | 0.7000 | +0.0333 |
| Hit@3 | 0.8333 | 0.8667 | +0.0333 |
| Hit@5 | 0.8667 | 0.8667 | +0.0000 |
| MRR | 0.7417 | 0.7722 | +0.0306 |
| nDCG@3 | 0.7377 | 0.7737 | +0.0360 |
| nDCG@5 | 0.7465 | 0.7737 | +0.0272 |
| Recall@5 | 0.8167 | 0.8167 | +0.0000 |
| Precision@5 | 0.1933 | 0.1933 | +0.0000 |

There were no Top-5 retrieval regressions or recoveries: both modes had the
same 26/30 Top-5 hits. Qwen improved the first relevant rank on four cases
(`rag_gpu_stale`, `rag_gpu_exclusive`, `rag_soft_preempt_reason`,
`rag_draining_meaning`) and regressed rank on one grounding case
(`rag_grounding_live`), while preserving Top-5 recall. The result supports
`RERANKER_NOT_REQUIRED` for the current corpus. It does not make Qwen the
default: offline local hashing remains the default, and Qwen remains an
explicit optional mode.

This validates embedding retrieval only, not the separately configured Qwen
chat/Agent provider or any external platform WRITE. No reranker was added.

The evaluator now normalizes context identities as `(source/section,
chunk_index)`: an explicit `source#section::chunkN` label matches only the
same chunk index, while an unqualified `source#section` label retains its
existing chunk-0 semantics. The canonical 30-case metrics above were rerun
and remained unchanged.
