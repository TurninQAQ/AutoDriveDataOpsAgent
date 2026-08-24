# Structured model provider

`providers/http_structured.py` implements an OpenAI-compatible structured
response adapter and `providers/qwen.py` supplies the Qwen/DashScope binding.
The request carries three explicitly separated channels:

1. Runtime Structured Context (authoritative deterministic state);
2. Operating Guidance (advisory principles);
3. Semantic Observation Context (`UNTRUSTED_EXTERNAL_DATA`).

The model returns one JSON AgentDecision proposal. It is parsed and then sent
through the existing `AgentDecisionIngressValidator`; malformed JSON/schema,
unknown variants, invalid goals, and invalid tool arguments become bounded
decision rejection. Transport timeout, rate limit, 5xx, and network failures
use explicit finite retry/timeouts and become provider-unavailable only after
the retry budget.

Credentials are referenced by environment-variable name. Secrets are never
placed in prompts, telemetry, event payloads, or logs. Tests use `httpx` local
MockTransport only; no paid model call is part of the regression suite.

## External smoke status

The production endpoint is configured by `AUTODRIVE_PROVIDER_ENDPOINT` and the
secret is supplied through the environment variable named by
`AUTODRIVE_PROVIDER_API_KEY_ENV` (normally `DASHSCOPE_API_KEY`). On 2026-08-24
the real validation process loaded the existing secret file only into its
process environment and used:

```text
provider: qwen
model: qwen-plus-2025-07-28
endpoint class: shared Beijing DashScope OpenAI-compatible
platform: localhost V2 in-process mock/simulated gateway
```

The production `QwenProvider` sent real requests to Bailian, parsed structured
AgentDecision proposals, and the Runtime/DecisionIngress boundary accepted
valid READ proposals. A clean single-READ run completed
`SINGLE_TOOL_CALL(get_gpu_pool) -> evidence -> FINAL_CANDIDATE -> CompletionGate`.
A combined run completed two `READ_TOOL_BATCH` decisions covering
`get_gpu_pool`, `get_queue_state`, and `search_knowledge`, followed by a
CompletionGate-approved final candidate. Queue results were normalized at the
V2 platform boundary from the canonical queue-file shape (`active: null` or an
active object) into the strict platform queue result contract.

The model emitted some malformed proposals during bounded recovery. They were
rejected by the existing typed Provider/DecisionIngress path; no parser bypass,
WRITE approval, or mutation occurred. The prompt was tightened only to state
the already-required initial GoalDescriptor and exact goal field rules. The
local adapter suite independently covers valid structured output, malformed
output, timeout, 429, 5xx, network failure, and bounded typed failure behavior.
