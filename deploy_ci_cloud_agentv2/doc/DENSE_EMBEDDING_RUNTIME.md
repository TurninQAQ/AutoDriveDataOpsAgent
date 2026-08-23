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

On 2026-08-24, real Qwen `qwen3.7-text-embedding` document and query smokes
both returned one normalized finite 1024-dimensional vector. The repository
knowledge corpus sidecar completed with 443 vectors for 30 documents. The
fixed five-case Golden Set measured hash Top-5 `1.00` / MRR `0.80` and Qwen
dense Top-5 `1.00` / MRR `0.90`; queue congestion improved from rank 2 to 1.

This validates retrieval only, not the separately configured Qwen chat/Agent
provider or any external platform WRITE. `RERANKER_NOT_REQUIRED` for this V2
phase; no reranker was added.
